import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        #self.model = model
        # self.example_input_array = torch.zeros(3, 224,224)  # optional

        if hasattr(model, "depth_est_model"):
            self.depth_est_model = model.depth_est_model

        if hasattr(model, "model"):
            model = model.model
            
        self.encoder = model.encoder
        self.decoder = model.decoder

    # def forward(self, x):
    #     return self.model(x)

def model_summary(model, max_depth=1):
    module_for_summary = LitModel(model)
    model_summary = ModelSummary(module_for_summary, max_depth=max_depth)
    print(model_summary)
