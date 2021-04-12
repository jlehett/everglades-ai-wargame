from utils.Statistics import AgentStatistics
from agents.Smart_State.render_smart_state import render_charts

# Create and load the statistics
SAVED_STATS_PATH = 'saved-stats/smart_state_newton'
stats = AgentStatistics()
stats.load_stats(SAVED_STATS_PATH)

# Render the charts
render_charts(stats)