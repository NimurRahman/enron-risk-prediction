# dashboard/tabs/__init__.py
from . import tab_activity
from . import tab_risk_over_time
from . import tab_top_risky
from . import tab_feature_importance
from . import tab_predictions
from . import tab_insights
from . import tab_ai_agent
from . import tab_simulator
from . import tab_clustering

__all__ = [
    'tab_activity',
    'tab_risk_over_time',
    'tab_top_risky',
    'tab_feature_importance',
    'tab_predictions',
    'tab_insights',
    'tab_ai_agent',
    'tab_simulator',
    'tab_clustering',
]