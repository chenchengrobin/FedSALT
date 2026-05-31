from torch.utils.data import Subset
import os

from src.utils.active_select import active_select
from src.utils.similarity import get_similarity

from src.training.evaluation import evaluate
from src.training.train import train_model

from src.loss.distillation import distillation_loss

from src.training.model import *
from src.loss.hybrid import *
from src.dataset import *
from src.config import *

class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = teacher(in_channels=3).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dataset = None
        self.dataloader = None
        self.logit = None
        self.similarity = None

    def local_train(self, test_dataset):
        train_model(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                criterion=self.criterion,
                early_stop_patience=EARLY_STOP_PATIENCE,
                need_return=False,
                )

        test_result = evaluate(model=self.model, dataset=test_dataset, mode='simple')
        print(f"Client {self.client_id} - Test Accuracy: {test_result[0]:.4f}")



class Server:
    def __init__(self):
        self.criterion = distillation_loss(reduction='mean', temperature=TEMPERATURE, alpha=0)
        self.model = None
        self.optimizer = None

        self.test_dataset = None
        self.test_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.dataset = None
        self.label = None
        self.federated_dataset = None
        self.federated_logit = None

        self.clients = []

    def init_server(self):
        self.model = student(in_channels=3).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # information extract
    def _generate_knowledge(self, mode, selected_indices=None):
        if selected_indices is None:
            selected_clients = self.clients
        else:
            selected_clients = [self.clients[i] for i in selected_indices]

        for client in selected_clients:
            model = client.model
            logit, global_feature, _ = evaluate(model=model, dataset=self.dataset, mode='extract')
            _, client_feature, _ = evaluate(model=model, dataset=client.dataset, mode='extract')
            client.logit = logit
            client.similarity = get_similarity(global_feature, client_feature, mode=mode)

    def _local_train(self, selected_indices=None):
        if selected_indices is None:
            selected_clients = self.clients
        else:
            selected_clients = [self.clients[i] for i in selected_indices]

        for client in selected_clients:
            client.local_train(test_dataset=self.test_dataset)


    def train(self):
        print(f"client number: {CLIENT_NUMBER}, client labeled number: {CLIENT_LABELED_NUMBER}, client pre-labeled number: {CLIENT_PRE_LABELED_NUMBER}", f"similarity mode: {SIMILARITY_MODE}")

        global_dataset = get_global_dataset('MNIST')

        # load train dataset
        train_dataset = global_dataset.get_train_dataset()
        self.distributor = data_distributor(train_dataset)

        # load test
        self.test_dataset = get_data_all_label(dataset=global_dataset.get_test_dataset(), number=TEST_NUMBER)     

        # load client dataset
        # select some clients to add gaussian noise randomly
        self.clients = [Client(client_id=i) for i in range(CLIENT_NUMBER)]
        for client in self.clients:
            label_list = np.arange(CLASS)
            labeled_list = np.random.choice(label_list, CLIENT_LABELED_NUMBER, replace=False)
            client.dataset, _ = self.distributor.distribute_client(number=CLIENT_PRE_LABELED_NUMBER, labels=labeled_list, balance=False)
            client.dataloader = get_dataloader(client.dataset, shuffle=True, drop_last=True)

        server_indices, server_labels = self.distributor.distribute_server(number=SERVER_NUMBER, balance=False)
        self.dataset = Subset(train_dataset, server_indices)
        self.dataloader = get_dataloader(self.dataset, shuffle=True, drop_last=True)
        self.label = server_labels
        print(f"client dataset has been loaded")

            
        self.init_server()
        self.federated_dataset = None
        self.federated_logit = None

        # extract clients' knowledge
        self._generate_knowledge(mode=SIMILARITY_MODE)

        # aggregate logit by active select and create federated dataset
        self.federated_logit = active_select(clients=self.clients, top_k=TOP_K)
        self.federated_dataset = create_dataset(self.dataset, self.federated_logit)
        federated_dataloader = get_dataloader(self.federated_dataset, shuffle=True, drop_last=False)

        # train server model
        train_model(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=federated_dataloader,
            criterion=self.criterion,
            early_stop_patience=EARLY_STOP_PATIENCE,
            verbose=True)

        # test server model
        student_result = evaluate(model=self.model, dataset=self.test_dataset, mode='simple')
        print(f"FedSALT - Test Accuracy: {student_result[0]:.4f}")

        # sove result
        save_dir = "result/one-shot"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(student_result, f"{save_dir}/{CLIENT_NUMBER}_{CLIENT_LABELED_NUMBER}_{CLIENT_PRE_LABELED_NUMBER}_{SIMILARITY_MODE}.pth")



if __name__ == '__main__':
    try:
        server = Server()
        server.train()
    except KeyboardInterrupt:
        print("训练被手动中断")
