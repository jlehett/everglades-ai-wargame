from matplotlib import pyplot as plt
import numpy as np

def render_charts(stats):
        
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
    par5 = ax3.twinx()
    par3.spines["right"].set_position(("axes", 1.1))
    par4.spines["right"].set_position(("axes", 1.1))
    #########################

    ######################
    #   Cumulative Plot  #
    ######################
    ax1.set_ylim([0.0,1.0])
    fig.suptitle('Win rates')
    ax1.plot(np.arange(1, stats.n_episodes+1),stats.scores)
    ax1.set_ylabel('Cumulative win rate')
    ax1.yaxis.label.set_color('blue')
    par1.plot(np.arange(1,stats.n_episodes+1),stats.epsilons,color="green")
    par1.set_ylabel('Eps/Temp')
    par1.yaxis.label.set_color('green')
    par3.plot(np.arange(1,stats.n_episodes+1),stats.network_loss,color="orange",alpha=0.5)
    par3.set_ylabel('Loss')
    par3.yaxis.label.set_color('orange')
    #######################

    ##################################
    #   Average Per K Episodes Plot  #
    ##################################
    ax2.set_ylim([0.0,1.0])
    par2.plot(np.arange(1,stats.n_episodes+1),stats.epsilons,color="green")
    par2.set_ylabel('Eps/Temp')
    par2.yaxis.label.set_color('green')
    par4.plot(np.arange(1,stats.n_episodes+1),stats.network_loss,color="orange",alpha=0.5)
    par4.set_ylabel('Loss')
    par4.yaxis.label.set_color('orange')
    ax2.plot(np.arange(0, stats.n_episodes+1, stats.k),stats.short_term_scores)
    ax2.set_ylabel('Average win rate')
    ax2.yaxis.label.set_color('blue')

    par3.tick_params(axis='y', colors='orange')
    par4.tick_params(axis='y', colors="orange")
    ax2.set_xlabel('Episode #')
    #############################

    #########################
    #   Average Q Val Plot  #
    #########################
    ax3.plot(np.arange(0, stats.n_episodes),stats.q_values)
    ax3.set_ylabel('Average Q Values')
    ax3.yaxis.label.set_color('blue')
    ax3.set_xlabel('Episode #')
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

    #########