# documented_pipelines
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re
from pathlib import Path

LAUNCHER_PREFIX = "python mp-launch.py"

PIPELINE_PATTERN = re.compile(
    r"(?:`)?\s*(python mp-launch\.py\s+.*?)(?:`)?(?=\n\n|\n\s*\n|$)", re.DOTALL
)
HEADING_PATTERN = re.compile(r"^#{2,3} (.+)$", re.MULTILINE)


def preceding_heading(headings, position):
    heading = ""
    for start, text in headings:
        if start > position:
            break
        heading = text
    return heading


def pipelines_by_section(doc_path):
    content = Path(doc_path).read_text(encoding="utf-8")
    headings = [
        (match.start(), match.group(1).strip())
        for match in HEADING_PATTERN.finditer(content)
    ]
    sections = {}
    for match in PIPELINE_PATTERN.finditer(content):
        description = match.group(1).strip().strip("`")
        description = description.removeprefix(LAUNCHER_PREFIX).strip()
        heading = preceding_heading(headings, match.start())
        sections.setdefault(heading, []).append(description)
    return sections
