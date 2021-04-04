import matplotlib.pyplot as plt
import numpy as np

def graph(n_episodes, scores, epsilonVals, lossVals, k, short_term_scores, short_term_final_scores, qVals):
    #####################
    # Plot final charts #
    #####################
    fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2)

    #########################
    #   Epsilon Plotting    #
    #########################
    par1 = ax1.twinx()
    par3 = ax1.twinx()
    par2 = ax2.twinx() 
    par4 = ax2.twinx()
    par3.spines["right"].set_position(("axes", 1.1))
    par4.spines["right"].set_position(("axes", 1.1))
    #########################

    ######################
    #   Cumulative Plot  #
    ######################
    ax1.set_ylim([0.0,1.0])
    fig.suptitle('Win rates')
    ax1.plot(np.arange(1, n_episodes+1),scores)
    ax1.set_ylabel('Cumulative win rate')
    ax1.yaxis.label.set_color('blue')
    par1.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
    par1.set_ylabel('Eps/Temp')
    par1.yaxis.label.set_color('green')
    par3.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
    par3.set_ylabel('Loss')
    par3.yaxis.label.set_color('orange')
    #######################

    ##################################
    #   Average Per K Episodes Plot  #
    ##################################
    ax2.set_ylim([0.0,1.0])
    par2.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
    par2.set_ylabel('Eps/Temp')
    par2.yaxis.label.set_color('green')
    par4.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
    par4.set_ylabel('Loss')
    par4.yaxis.label.set_color('orange')
    ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
    ax2.set_ylabel('Average win rate')
    ax2.yaxis.label.set_color('blue')

    par3.tick_params(axis='y', colors='orange')
    par4.tick_params(axis='y', colors="orange")
    ax2.set_xlabel('Episode #')
    #############################

    #########################
    #   Average Reward Plot #
    #########################
    ax3.plot(np.arange(0, n_episodes+1,k),short_term_final_scores)
    ax3.set_ylabel('Average Final Scores')
    ax3.yaxis.label.set_color('blue')
    #########################

    #########################
    #   Average Q Val Plot  #
    #########################
    ax4.plot(np.arange(0, n_episodes),qVals)
    ax4.set_ylabel('Average Q Values')
    ax4.yaxis.label.set_color('blue')
    ax4.set_xlabel('Episode #')
    #########################

    fig.tight_layout(pad=2.0)
    plt.show()
    #########################
    #   Setup Loss Spines   #
    #########################
    for ax in [par3, par4]:
        ax.set_frame_on(True)
        ax.patch.set_visible(False)

        plt.setp(ax.spines.values(), visible=False)
        ax.spines["right"].set_visible(True)

    #########################