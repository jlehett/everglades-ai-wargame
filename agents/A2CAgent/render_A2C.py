from matplotlib import pyplot as plt
import numpy as np

def render_charts(stats):
    """
    Renders charts for PPO using stored statistical data
    @params stats The statistics class that stores the data to be plotted
    """
    #####################
    # Plot final charts #
    #####################
    fig, (ax1,ax2) = plt.subplots(2,1)

    #########################
    #   Epsilon Plotting    #
    #########################
    par1 = ax1.twinx()
    par2 = ax2.twinx()
    #########################

    ######################
    #   Cumulative Plot  #
    ######################
    fig.suptitle('Scores')
    ax1.plot(np.arange(1, stats.n_episodes+1),stats.scores)
    ax1.set_ylabel('Cumulative Scores')
    ax1.yaxis.label.set_color('blue')
    par1.plot(np.arange(1,stats.n_episodes+1),stats.network_loss,color="red",alpha=0.5)
    par1.set_ylabel('Loss')
    par1.yaxis.label.set_color('red')
    par1.set_ylim([-40, 40])
    #######################

    ##################################
    #   Average Per K Episodes Plot  #
    ##################################
    par2.plot(np.arange(1,stats.n_episodes+1),stats.network_loss,color="red",alpha=0.5)
    par2.set_ylabel('Loss')
    par2.yaxis.label.set_color('red')
    ax2.plot(np.arange(0, stats.n_episodes+1, stats.k),stats.short_term_scores)
    par2.set_ylim([-40, 40])

    # Create a Line of Best Fit
    x = np.arange(1, stats.n_episodes+1, stats.k)
    y = stats.short_term_scores[1:]
    m, b = np.polyfit(x, y, 1)

    ax2.plot(x, m*x + b)

    ax2.set_ylabel('Average Scores')
    ax2.yaxis.label.set_color('blue')
    ax2.set_xlabel('Episode #')
    #############################
    
    plt.show()
    #########
