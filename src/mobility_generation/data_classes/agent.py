from dataclasses import dataclass

from src.survey_processing.dataclasses.agenda import Agenda, AgendaSegment
from src.mobility_generation.data_classes.route import Route, RouteElement


@dataclass(slots=True)
class Agent:
    """
    An agent captures information about their agenda and route. An agent furthermore has a state that remembers the
    segment in the agent's agenda that is currently traversed.
    """
    agenda: Agenda
    history: Route  # previous route the agent has taken (consists of cell ids)
    current_segment: AgendaSegment
    current_segment_idx: int
    start_cell_idx: int

    @classmethod
    def from_agenda(cls, agenda: Agenda) -> "Agent":
        """
        Create a new agent with the given agenda.
        """
        # initialize agent with first agenda segment and empty route
        return cls(
            agenda=agenda,
            history=Route(),
            current_segment=agenda.get_agenda_segment(0),   # get first agenda segment
            current_segment_idx=0,
            start_cell_idx=-1  # defaults to a non-valid cell id
        )

    # def __init__(self, agenda: Agenda):
    #     """
    #     Create a new agent with the given agenda.
    #     """
    #     self.agenda = Agenda
    #     self.history = Route()
    #     self.current_segment = agenda.get_agenda_segment(0) # initialize with first agenda segment
    #     self.current_segment_idx = 0
    #     self.start_cell_id = -1

    @property
    def is_active(self) -> bool:
        """Return true if the agent's state refers to an agenda segment."""
        return self.current_segment_idx < self.agenda.number_of_segments

    def next_segment(self) -> None:
        """Increment the agent's state to the next segment."""
        self.current_segment_idx += 1
        if not self.is_active:
            raise StopIteration("traversed all segments")
        self.current_segment = self.agenda.get_agenda_segment(self.current_segment_idx)
        return

    def append_route_element(self, element: RouteElement) -> None:
        """Append the given route element to the agent's history."""
        self.history.append_element(element=element)
        return
