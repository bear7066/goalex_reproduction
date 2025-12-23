from typing import List, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ClusterProblem:
    goal: str
    texts: List[str]
    example_descriptions: Optional[List[str]] = None


@dataclass_json
@dataclass
class ClusterProblemLabel:
    class_descriptions: List[str]
    labels: List[int]