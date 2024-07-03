import matplotlib.pyplot as plt

def plot(scores, mean_scores):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(mean_scores)
    plt.title('Mean Rewards (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Mean Total Reward')

    plt.tight_layout()
    plt.show()
