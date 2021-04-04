from matplotlib import pyplot as plt
import numpy as np

def render_charts(stats):
        
    #####################
    # Plot final charts #
    #####################
    fig, axs = plt.subplots(2)

    #########################
    #   Epsilon Plotting    #
    #########################
    par1 = axs[0].twinx()
    par3 = axs[0].twinx()
    par2 = axs[1].twinx()
    par4 = axs[1].twinx()
    par3.spines["right"].set_position(("axes", 1.1))
    par4.spines["right"].set_position(("axes", 1.1))
    #########################

    ######################
    #   Cumulative Plot  #
    ######################
    axs[0].set_ylim([0.0,1.0])
    fig.suptitle('Win rates')
    axs[0].plot(np.arange(1, stats.n_episodes+1),stats.scores)
    axs[0].set_ylabel('Cumulative win rate')
    axs[0].yaxis.label.set_color('blue')
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
    axs[1].set_ylim([0.0,1.0])
    par2.plot(np.arange(1,stats.n_episodes+1),stats.epsilons,color="green")
    par2.set_ylabel('Eps/Temp')
    par2.yaxis.label.set_color('green')
    par4.plot(np.arange(1,stats.n_episodes+1),stats.network_loss,color="orange",alpha=0.5)
    par4.set_ylabel('Loss')
    par4.yaxis.label.set_color('orange')
    axs[1].plot(np.arange(0, stats.n_episodes+1, stats.k),stats.short_term_scores)
    axs[1].set_ylabel('Average win rate')
    axs[1].yaxis.label.set_color('blue')

    par3.tick_params(axis='y', colors='orange')
    par4.tick_params(axis='y', colors="orange")
    axs[1].set_xlabel('Episode #')
    #############################

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