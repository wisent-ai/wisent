"""Blind A/B Test Game component."""
import gradio as gr
import random
from typing import Dict, Optional, Tuple
from tests.EVALOOP.core.data_manager import DataManager


class BlindTestGame:
    """Manages the Blind A/B Test Game for detecting steered responses."""

    def __init__(self, data_manager: DataManager, total_rounds: int = 10):
        """
        Initialize BlindTestGame.

        Args:
            data_manager: DataManager instance
            total_rounds: Number of rounds in the game
        """
        self.data_manager = data_manager
        self.total_rounds = total_rounds
        self.components = {}
        self._create_components()

    def _create_components(self):
        """Create game UI components."""
        # File selector
        gr.Markdown("""
        ## Can you identify which response is steered?

        **Test your ability to distinguish baseline from steered responses.**

        Play 10 rounds! In each round, you'll see two responses (A and B) to the same question.
        One is from the baseline model, the other is steered. Try to guess which one is steered.
        """)

        with gr.Row():
            from tests.EVALOOP.ui.components.controls import FileSelector
            file_selector = FileSelector(self.data_manager, "scores")
            self.components['file_dropdown'] = file_selector.get_component()

        self.components['start_btn'] = gr.Button("Start Game", variant="primary")
        self.components['progress'] = gr.Markdown("**Score:** 0/0")
        self.components['question'] = gr.Markdown("")

        # Responses
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Response A")
                self.components['response_a'] = gr.Textbox(
                    label="", lines=15, max_lines=20, interactive=False
                )

            with gr.Column():
                gr.Markdown("### Response B")
                self.components['response_b'] = gr.Textbox(
                    label="", lines=15, max_lines=20, interactive=False
                )

        # Guess buttons
        with gr.Row():
            self.components['guess_a_btn'] = gr.Button("Response A is Steered", visible=False)
            self.components['guess_b_btn'] = gr.Button("Response B is Steered", visible=False)

        self.components['result'] = gr.Markdown("")

        # Game state
        self.components['state'] = gr.State(value=self._get_initial_state())

    def _get_initial_state(self) -> Dict:
        """Get initial game state."""
        return {
            "file_path": None,
            "current_entry": None,
            "is_left_steered": None,
            "round": 0,
            "score": 0,
            "total_rounds": self.total_rounds,
            "game_active": False
        }

    def start_game(self, file_path: str, state: Dict) -> Tuple:
        """
        Start a new game.

        Args:
            file_path: Path to scores file
            state: Game state dictionary

        Returns:
            Tuple of component updates
        """
        # Reset state
        state["file_path"] = file_path
        state["round"] = 1
        state["score"] = 0
        state["game_active"] = True

        # Load first question
        return self._load_next_question(state)

    def _load_next_question(self, state: Dict) -> Tuple:
        """Load the next question pair."""
        try:
            entry, _ = self.data_manager.get_random_entry(state["file_path"])

            if entry is None:
                return (
                    "No entries found.", "", "", "", "",
                    gr.Button(visible=True),
                    gr.Button(visible=False),
                    gr.Button(visible=False),
                    state
                )

            # Randomly decide if steered is on left or right
            is_left_steered = random.choice([True, False])

            # Store state
            state["current_entry"] = entry
            state["is_left_steered"] = is_left_steered

            # Prepare displays
            question = f"**Round {state['round']}/{state['total_rounds']}**\n\n**Question:** {entry['question']}"

            if is_left_steered:
                response_a = entry['steered_response']
                response_b = entry['baseline_response']
            else:
                response_a = entry['baseline_response']
                response_b = entry['steered_response']

            progress = f"**Score:** {state['score']}/{state['round']-1}" if state['round'] > 1 else "**Score:** 0/0"

            # Hide start button, show guess buttons
            return (
                question, response_a, response_b, "", progress,
                gr.Button(visible=False),
                gr.Button(visible=True),
                gr.Button(visible=True),
                state
            )

        except Exception as e:
            return (
                f"Error: {e}", "", "", "", "",
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                state
            )

    def check_guess(self, choice: str, file_path: str, state: Dict) -> Tuple:
        """
        Check user's guess and load next question or end game.

        Args:
            choice: User's choice ("A" or "B")
            file_path: Current file path
            state: Game state

        Returns:
            Tuple of component updates
        """
        if not state["game_active"]:
            return (
                "Game not started!", "", "", "", "",
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                state
            )

        is_left_steered = state["is_left_steered"]

        # Check if correct
        is_correct = (choice == "A" and is_left_steered) or (choice == "B" and not is_left_steered)

        if is_correct:
            state["score"] += 1
            result = "# ✅ Correct!\n\n"
        else:
            result = "# ❌ Incorrect\n\n"

        # Show which was which
        if is_left_steered:
            result += "Response A was **steered**, Response B was **baseline**."
        else:
            result += "Response A was **baseline**, Response B was **steered**."

        # Check if game is over
        if state["round"] >= state["total_rounds"]:
            # Game over
            state["game_active"] = False
            final_score = state["score"]
            total = state["total_rounds"]
            percentage = (final_score / total) * 100

            result += f"\n\n# 🎮 Game Over!\n\n**Final Score: {final_score}/{total} ({percentage:.1f}%)**"
            progress = f"**Final Score:** {final_score}/{total}"

            # Clear question and responses, show start button, hide guess buttons
            return (
                result, "", "", "", progress,
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                state
            )
        else:
            # Load next round
            state["round"] += 1

            # Get next question from current file path
            entry, _ = self.data_manager.get_random_entry(file_path)
            is_left_steered = random.choice([True, False])

            state["current_entry"] = entry
            state["is_left_steered"] = is_left_steered

            question = f"**Round {state['round']}/{state['total_rounds']}**\n\n**Question:** {entry['question']}"

            if is_left_steered:
                response_a = entry['steered_response']
                response_b = entry['baseline_response']
            else:
                response_a = entry['baseline_response']
                response_b = entry['steered_response']

            progress = f"**Score:** {state['score']}/{state['round']-1}"

            # Keep guess buttons visible
            return (
                result, question, response_a, response_b, progress,
                gr.Button(visible=False),
                gr.Button(visible=True),
                gr.Button(visible=True),
                state
            )

    def wire_events(self):
        """Wire up all event handlers."""
        # Start game returns: question, response_a, response_b, result, progress, start_btn, guess_a_btn, guess_b_btn, state
        start_outputs = [
            self.components['question'],
            self.components['response_a'],
            self.components['response_b'],
            self.components['result'],
            self.components['progress'],
            self.components['start_btn'],
            self.components['guess_a_btn'],
            self.components['guess_b_btn'],
            self.components['state']
        ]

        # Guess functions return: result, question, response_a, response_b, progress, start_btn, guess_a_btn, guess_b_btn, state
        guess_outputs = [
            self.components['result'],
            self.components['question'],
            self.components['response_a'],
            self.components['response_b'],
            self.components['progress'],
            self.components['start_btn'],
            self.components['guess_a_btn'],
            self.components['guess_b_btn'],
            self.components['state']
        ]

        # Start game button
        self.components['start_btn'].click(
            fn=self.start_game,
            inputs=[self.components['file_dropdown'], self.components['state']],
            outputs=start_outputs
        )

        # Guess buttons
        self.components['guess_a_btn'].click(
            fn=lambda fp, state: self.check_guess("A", fp, state),
            inputs=[self.components['file_dropdown'], self.components['state']],
            outputs=guess_outputs
        )

        self.components['guess_b_btn'].click(
            fn=lambda fp, state: self.check_guess("B", fp, state),
            inputs=[self.components['file_dropdown'], self.components['state']],
            outputs=guess_outputs
        )
