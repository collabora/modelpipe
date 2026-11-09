import os
import sys
from pathlib import Path

# the wheel nests the element modules here, a checkout puts them on sys.path already
INSTALLED_ELEMENT_MODULES = Path(__file__).parent / "plugins" / "python"
if INSTALLED_ELEMENT_MODULES.is_dir():
    sys.path.insert(0, str(INSTALLED_ELEMENT_MODULES))


def launch():
    import mp_launch

    return mp_launch.main()


def mcp():
    # mp_mcp imports gi at module scope
    if os.environ.get("MP_BACKEND", "gst").lower() == "g2g":
        import mp_mcp_g2g

        sys.exit(mp_mcp_g2g.main())

    import mp_mcp

    mp_mcp.main()


def engines():
    import mp_engines

    mp_engines.main()
