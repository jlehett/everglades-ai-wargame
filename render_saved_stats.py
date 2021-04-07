from utils.Statistics import AgentStatistics
from agents.PPO.render_ppo import render_charts

# Create and load the statistics
SAVED_STATS_PATH = 'saved-stats/rppo_new'
stats = AgentStatistics()
stats.load_stats(SAVED_STATS_PATH)

# Render the charts
render_charts(stats)