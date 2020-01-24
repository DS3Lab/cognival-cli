import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_handler(title, history, log, output_dir):
    '''
    Renders training history (loss and validation loss) as
    PNG plot.

    :param title: Plot title and filename for output file
    :param history: History dictionary
    :param log: Experiment log
    :param output_dir: Output path
    '''
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(Path(output_dir) / '{}.png'.format(log["wordEmbedding"]))
    plt.clf()

    return