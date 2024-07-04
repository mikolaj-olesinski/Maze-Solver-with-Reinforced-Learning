import matplotlib.pyplot as plt
from IPython import display
import contextlib
import io
import numpy as np

plt.ion()

def plot(scores, mean_scores, filepath=None):
    with contextlib.redirect_stdout(io.StringIO()):  # Suppress print output
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.plot([np.mean(scores[-100:])]*len(scores))
        plt.ylim(ymin=0)
        plt.legend(['Score', 'Mean Score', 'Mean Last 100'])
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.text(len(scores)-1, np.mean(scores[-100:]), str(np.mean(scores[-100:])))
        plt.show(block=False)
        
        if filepath:
            plt.savefig(filepath)

        plt.pause(.1)

