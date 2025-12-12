import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from demo.agent_cli import build_parser


def test_agent_cli_parser_builds():
    p = build_parser()
    args = p.parse_args(["--question", "q"])
    assert args.question == "q"

