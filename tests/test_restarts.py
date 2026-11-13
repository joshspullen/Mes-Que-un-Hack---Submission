"""Unit test for mqh/restarts.py on a made-up tagging file. No real data involved.

    python tests/test_restarts.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mqh.restarts import RestartConfig, dead_mask_from_tags, restart_frames  # noqa: E402
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
    print("OK  restarts: 2 restarts kept out of 4 tags, mask covers {} frames".format(dead.sum()))


if __name__ == "__main__":
    main()
