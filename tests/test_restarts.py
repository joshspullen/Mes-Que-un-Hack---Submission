"""Unit test for mqh/restarts.py on a made-up tagging file. No real data involved.

    python tests/test_restarts.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from mqh.restarts import (RestartConfig, dead_mask_from_tags, push_from_stoppages,  # noqa: E402
                          restart_frames, restart_windows, stoppages, team_sign)
from starter.load_smart_tagging import load  # noqa: E402

XML = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>100.0</start><end>104.0</end><code>SET PIECES</code>
  <label><group>Type</group><text>Throw-in</text></label>
  <label><group>Team</group><text>N/A</text></label></instance>
<instance><ID>2</ID><start>200.0</start><end>210.0</end><code>BUILD UP</code>
  <label><group>Team</group><text>Barcelona</text></label></instance>
<instance><ID>3</ID><start>300.0</start><end>306.0</end><code>Set Pieces</code>
  <label><group>Type</group><text>CORNER KICK</text></label></instance>
<instance><ID>4</ID><start>400.0</start><end>404.0</end><code>SET PIECES</code>
  <label><group>Type</group><text>Something New</text></label></instance>
</ALL_INSTANCES></file>
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "tactical")
        os.makedirs(folder)
        with open(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), "w") as handle:
            handle.write(XML)
        events = load(data_dir=tmp)

    cfg = RestartConfig(pre_dead_s={"throw-in": 8.0, "corner kick": 15.0})
    # build-up and unknown types are ignored; case does not matter
    assert restart_frames(events, cfg) == [("throw-in", 2550), ("corner kick", 7575)], \
        restart_frames(events, cfg)

    dead = dead_mask_from_tags(events, 20000, cfg)
    assert dead[2550 - 200:2550].all() and not dead[2550] and not dead[2550 - 201]
    assert dead[7575 - 375:7575].all() and not dead[7575]
    assert dead.sum() == 200 + 375
    # v2: a tracking stoppage near the tag replaces the fixed duration. Throw-in tag = frames
    # 2500-2600; the tracking saw the ball dead on 2300-2400 and 2420-2580 (hole of 20 frames).
    track = np.zeros(20000, dtype=bool)
    track[2300:2400] = track[2420:2580] = True
    track[9000:9100] = True                                   # far from any tag: ignored
    v2 = dead_mask_from_tags(events, 20000, cfg, track_dead=track)
    assert v2[2300:2580].all() and not v2[2299] and not v2[2580], "tracking anchor / bridge"
    assert v2[7575 - 375:7575].all() and not v2[7575], "no tracking stoppage -> v1 fallback"
    assert not v2[9000:9100].any()
    assert v2.sum() == 280 + 375
    # the restart may sit up to tolerance_s outside the tag, not further
    late = np.zeros(20000, dtype=bool)
    late[2550:2900] = True
    assert dead_mask_from_tags(events, 20000, cfg, track_dead=late)[2550:2650].all()
    assert not dead_mask_from_tags(events, 20000, cfg, track_dead=late)[2650:7000].any()

    # restart team: named teams push, throw-ins / unknown names do not
    assert (team_sign("FC Barcelona"), team_sign("Athletic Club"), team_sign("N/A"), team_sign(None))         == (1, -1, 0, 0)
    wins = restart_windows(events)
    assert [w[3] for w in wins] == [0, 0]
    stops = stoppages([("corner kick", 7500, 7650, -1)], 20000, cfg)
    push = push_from_stoppages(stops, 20000, RestartConfig(push=0.5, push_s=2.0))
    assert (push[7575:7625] == -0.5).all() and push[7574] == 0 and push[7625] == 0
    assert not push_from_stoppages(stops, 20000, RestartConfig(push=0.0)).any()

    print("OK  restarts: 2 restarts kept out of 4 tags, mask covers {} frames".format(dead.sum()))


if __name__ == "__main__":
    main()
