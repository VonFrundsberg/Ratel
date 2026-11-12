"""RATEL: Rodent Activity Tracking and Ethogram Lab.

Every tool (RATEL's OpenCV tracker, ezTrack, AnimApp, DeepLabCut or manual
scoring) ends up in the same long table::

    video_id, tool, metric, zone, value

so that ``behaviour.compare`` can measure how well they agree.
"""

__version__ = "0.2.0"
