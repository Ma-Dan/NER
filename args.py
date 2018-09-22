import argparse
from utils import str2bool

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese Word NER task with gensim')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=False, help='update embedding during training')
parser.add_argument('--embedding_dim', type=int, default=400, help='word embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1537601731', help='model for test and demo')

parser.add_argument('--word_embedding_file', type=str, default='resource/wiki.zh.vec', help='Pretrained word embeddings.')
parser.add_argument('--src_file', type=str, default='resource/source.txt', help='Training data.')
parser.add_argument('--tgt_file', type=str, default='resource/target.txt', help='Labels.')
parser.add_argument('--src_vocab_file', type=str, default='resource/source_vocab.txt', help='source vocabulary.')
parser.add_argument('--tgt_vocab_file', type=str, default='resource/target_vocab.txt', help='target vocabulary.')
parser.add_argument('--max_sequence', type=int, default=100, help='max sequence length.')

args = parser.parse_args()
