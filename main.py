from argument_parser import argument_parser
from models import Trainer
from utils import table_printer

args = argument_parser()
table_printer(args)

# Initialize a trainer
trainer = Trainer(args)

# load edgelist and split into training, test and validation set
trainer.setup_features()

# Setup a Biological Network Embedding model
trainer.setup_model()
trainer.setup_training_data()
trainer.fit()
trainer.evaluate()
