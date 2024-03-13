
import torch
import torch.nn as nn
import torchvision
from src.model.models import get_models
from tqdm import tqdm



class NormalTrain():

    def __init__(self,args,train_loader,test_loader):
        self.NUMEPOCHS = args.epochs
        self.DEVICE = args.device
        self.lr = args.learningrate

        self.trainloader = train_loader
        self.testloader = test_loader
        self.models = get_models()


       

    def train(self):
        for model in self.models:
            model = self.train_model(model)
            y_pred, y_test = self.evaluate(model["model"])
            self.save_results(y_pred,y_test)
        return y_pred,y_test
    



    def evaluate(self,model,test_loader):
        model.eval()
        model.to(torch.device("cpu"))
        with torch.no_grad():
            all_preds = []
            all_test = []
            for i in test_loader:
                sample, label = i
                logits = model(sample)
                y_pred = (logits > 0.5).int()
                all_preds.extend(y_pred.cpu().numpy())
                all_test.extend(label.cpu().numpy())
        return all_preds,all_test
    
    

    def train_model(self,models):
        model = models["model"]
        loss = []
        
        model.to(self.DEVICE)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)

        for epoch in tqdm(range(self.NUMEPOCHS)):
            running_loss = 0.0
            for sample in self.trainloader:
                image, label = sample
                logits = model(image)
                l = criterion(logits,label)
                loss.append(l)
                running_loss +=l
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            tqdm.write(f"Epoch {epoch}/{self.NUMEPOCHS}  Loss : {running_loss}")
            running_loss = 0.0
        models["loss"] = loss
        models["model"] = model
        return models


                



    



