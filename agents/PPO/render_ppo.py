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
    fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2)

    #########################
    #   Epsilon Plotting    #
    #########################
    par1 = ax1.twinx()
    par2 = ax2.twinx()
    par3 = ax1.twinx()
    par4 = ax2.twinx()
    #########################

    ######################
    #   Cumulative Plot  #
    ######################
    fig.suptitle('Scores')
    ax1.plot(np.arange(1, stats.n_episodes+1),stats.scores)
    ax1.set_ylabel('Cumulative Scores')
    ax1.yaxis.label.set_color('blue')
    par1.plot(np.arange(1,stats.n_episodes+1),stats.lossVals,color="red",alpha=0.5)
    par1.set_ylabel('Loss')
    par1.yaxis.label.set_color('red')
    #######################

    ##################################
    #   Average Per K Episodes Plot  #
    ##################################
    par2.plot(np.arange(1,stats.n_episodes+1),stats.lossVals,color="red",alpha=0.5)
    par2.set_ylabel('Loss')
    par2.yaxis.label.set_color('red')
    ax2.plot(np.arange(0, stats.n_episodes+1, stats.k),stats.short_term_scores)
    ax2.set_ylabel('Average Scores')
    ax2.yaxis.label.set_color('blue')
    ax2.set_xlabel('Episode #')
    #############################

    ##################################
    #   Actor Loss Plot              #
    ##################################
    par3.plot(np.arange(1,stats.n_episodes+1),stats.lossVals,color="red",alpha=0.5)
    par3.set_ylabel('Loss')
    par3.yaxis.label.set_color('red')
    ax3.plot(np.arange(1, stats.n_episodes+1),stats.actorLossVals)
    ax3.set_ylabel('Actor Loss')
    ax3.yaxis.label.set_color('blue')
    ax3.set_xlabel('Episode #')
    ##################################

    ##################################
    #   Critic Loss Plot             #
    ##################################
    par4.plot(np.arange(1,stats.n_episodes+1),stats.lossVals,color="red",alpha=0.5)
    par4.set_ylabel('Loss')
    par4.yaxis.label.set_color('red')
    ax4.plot(np.arange(1, stats.n_episodes+1),stats.criticLossVals)
    ax4.set_ylabel('Critic Loss')
    ax4.yaxis.label.set_color('blue')
    ax4.set_xlabel('Episode #')
    ##################################

    plt.show()
    #########