#!/usr/bin/env python3
"""Generate synthetic PDF seed fixtures for Catan game documents.

Creates placeholder PDF rulebooks with original synthetic content that covers
the same mechanical topics as the real Catan rulebooks. These are NOT copies
of copyrighted material  -- they are original summaries written for testing
the document ingestion pipeline, KG construction, and AI query validation.

Usage:
    python scripts/generate_seed_pdfs.py

Output:
    src/tabletop_oracle/fixtures/seed/games/catan/documents/catan-base-rules.pdf
    src/tabletop_oracle/fixtures/seed/games/catan/documents/catan-seafarers-rules.pdf
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

DOCS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tabletop_oracle"
    / "fixtures"
    / "seed"
    / "games"
    / "catan"
    / "documents"
)

# ---------------------------------------------------------------------------
# Content  -- Catan Base Rules (synthetic, original text)
# ---------------------------------------------------------------------------

BASE_RULES_TITLE = "Catan Base Game Rules"
BASE_RULES_SUBTITLE = "Seed Fixture - Synthetic Reference Document (5th Edition)"

BASE_RULES_SECTIONS: list[tuple[str, str]] = [
    (
        "Overview",
        (
            "Catan is a multiplayer board game for 3 to 4 players designed by Klaus Teuber "
            "and first published in 1995 by Kosmos. Players take on the role of settlers "
            "establishing colonies on the island of Catan. The game combines resource "
            "management, trading, and strategic building. The objective is to be the first "
            "player to reach 10 victory points through a combination of settlements, "
            "cities, development cards, and special achievements."
        ),
    ),
    (
        "Components",
        (
            "The game includes 19 terrain hexes (4 forest, 4 pasture, 4 fields, "
            "3 hills, 3 mountains, 1 desert), 6 sea frame pieces, 18 circular number "
            "tokens (numbered 2 through 12, excluding 7), 95 resource cards (19 each of "
            "lumber, wool, grain, brick, and ore), 25 development cards (14 knight cards, "
            "5 victory point cards, 2 road building cards, 2 year of plenty cards, "
            "2 monopoly cards), 4 building costs reference cards, 2 special cards "
            "(Longest Road and Largest Army), 16 cities (4 per colour), "
            "20 settlements (5 per colour), 60 roads (15 per colour), "
            "1 robber pawn, and 2 dice."
        ),
    ),
    (
        "Setup",
        (
            "Arrange the 19 terrain hexes randomly within the sea frame to form the "
            "island. Place number tokens on each terrain hex (except the desert) in "
            "alphabetical order following the spiral pattern shown in the almanac. "
            "The desert hex receives the robber pawn. Each player selects a colour and "
            "takes 5 settlements, 4 cities, and 15 roads of that colour. Players take "
            "turns placing 1 settlement at any intersection of three hexes and 1 road "
            "adjacent to it. After all players have placed their first settlement and "
            "road, placement continues in reverse order for the second settlement and "
            "road. Each player collects 1 resource card for each terrain hex adjacent "
            "to their second settlement."
        ),
    ),
    (
        "Terrain and Resources",
        (
            "Each terrain hex produces a specific resource when its number is rolled. "
            "Forest hexes produce lumber. Pasture hexes produce wool. Fields produce "
            "grain. Hills produce brick. Mountains produce ore. The desert hex produces "
            "no resources. Resource cards are the currency of the game  -- they are spent "
            "to build roads, settlements, cities, and to buy development cards."
        ),
    ),
    (
        "Turn Structure",
        (
            "On each turn, a player performs three phases in order. First, the player "
            "rolls both dice and sums the result. Every player who has a settlement "
            "adjacent to a terrain hex matching the rolled number receives 1 resource "
            "card of that type. Players with a city adjacent to the hex receive 2 "
            "resource cards instead. Second, the player may trade resources with other "
            "players or with the bank (see Trading). Third, the player may build roads, "
            "settlements, or cities, or buy development cards by paying the required "
            "resources (see Building)."
        ),
    ),
    (
        "Trading",
        (
            "There are two forms of trading: domestic trade and maritime trade. "
            "In domestic trade, the active player may offer resources to any other player "
            "and negotiate freely. Only the active player may initiate trades. Other "
            "players may not trade among themselves. In maritime trade, any player may "
            "trade 4 identical resource cards of any type to the bank for 1 resource card "
            "of any other type. If a player has a settlement or city on a harbour, "
            "they may trade at a better rate. Generic harbours allow 3:1 trades of any "
            "resource. Specific harbours (e.g., lumber harbour) allow 2:1 trades of the "
            "named resource."
        ),
    ),
    (
        "Building",
        (
            "Players build by paying resource cards to the bank. A road costs 1 brick "
            "and 1 lumber. A settlement costs 1 brick, 1 lumber, 1 wool, and 1 grain. "
            "A city upgrade (replacing a settlement) costs 2 grain and 3 ore. "
            "A development card costs 1 wool, 1 grain, and 1 ore. Roads are placed along "
            "the edges between hexes and must connect to an existing road, settlement, or "
            "city belonging to the player. Settlements are placed at intersections where "
            "three hexes meet and must be connected to the player's road network. "
            "The distance rule requires that no two settlements (or cities) may be placed "
            "on adjacent intersections  -- there must be at least 2 edges between any two "
            "settlements. Cities replace existing settlements and produce double resources."
        ),
    ),
    (
        "Development Cards",
        (
            "Development cards are purchased for 1 wool, 1 grain, and 1 ore and drawn "
            "from the shuffled development card deck. A player may play at most 1 "
            "development card per turn, and may not play a card on the same turn it was "
            "purchased. Knight cards: when played, the player moves the robber to any "
            "terrain hex and steals 1 random resource card from a player with a "
            "settlement or city adjacent to that hex. Road Building: the player "
            "immediately places 2 roads at no cost. Year of Plenty: the player takes "
            "any 2 resource cards from the bank. Monopoly: the player names 1 resource "
            "type and all other players must give all their cards of that type to the "
            "player. Victory Point cards are kept hidden and revealed only when the "
            "player reaches 10 victory points."
        ),
    ),
    (
        "The Robber",
        (
            "When a player rolls a 7, no resources are produced. Instead, every player "
            "with more than 7 resource cards must discard half (rounded down). The player "
            "who rolled the 7 must then move the robber pawn to any terrain hex other than "
            "its current location. While the robber is on a hex, that hex does not produce "
            "resources when its number is rolled. After moving the robber, the player may "
            "steal 1 random resource card from any player who has a settlement or city "
            "adjacent to the robber's new hex."
        ),
    ),
    (
        "Special Cards",
        (
            "Longest Road: the first player to build a continuous road of at least 5 "
            "segments receives the Longest Road card, worth 2 victory points. If another "
            "player builds a longer continuous road, the card transfers to that player. "
            "Largest Army: the first player to play 3 knight development cards receives "
            "the Largest Army card, worth 2 victory points. If another player plays more "
            "knights, the card transfers to that player."
        ),
    ),
    (
        "Victory Conditions",
        (
            "The game ends immediately when a player reaches 10 or more victory points "
            "on their turn. Victory points are earned as follows: each settlement is "
            "worth 1 victory point, each city is worth 2 victory points, the Longest "
            "Road card is worth 2 victory points, the Largest Army card is worth 2 "
            "victory points, and each victory point development card is worth 1 victory "
            "point. A player may only win during their own turn. If a player reaches 10 "
            "points during another player's turn (e.g., due to Longest Road transfer), "
            "they do not win until their own next turn, when they must announce their "
            "victory."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Content  -- Catan Seafarers Expansion Rules (synthetic, original text)
# ---------------------------------------------------------------------------

SEAFARERS_TITLE = "Catan: Seafarers Expansion Rules"
SEAFARERS_SUBTITLE = "Seed Fixture - Synthetic Reference Document"

SEAFARERS_SECTIONS: list[tuple[str, str]] = [
    (
        "Overview",
        (
            "Catan: Seafarers is an expansion for the Catan base game, designed by "
            "Klaus Teuber and first published in 1997. It extends the game to include "
            "sea travel, ship building, and exploration of new islands. The expansion "
            "requires the Catan base game to play and supports 3 to 4 players (expandable "
            "to 5-6 with an additional extension). Seafarers introduces ships, the pirate, "
            "gold river hexes, and multiple island scenarios with varied victory point "
            "targets."
        ),
    ),
    (
        "New Components",
        (
            "Seafarers adds 60 ships (15 per colour), 1 pirate pawn, 6 frame pieces "
            "for extended sea borders, additional terrain hexes including gold river "
            "hexes, additional harbour tokens, and scenario booklets. Ships use the same "
            "colour pieces as roads and share the 15-piece limit with roads  -- a player "
            "has 15 total pieces that can be roads or ships."
        ),
    ),
    (
        "Ships",
        (
            "Ships are the maritime equivalent of roads. A ship costs 1 wool and 1 lumber "
            "to build. Ships are placed on edges between two sea hexes, or between a sea "
            "hex and a land hex (coastal edges). Ships connect to settlements, cities, "
            "and other ships to form shipping routes. A shipping route is a continuous "
            "chain of ships connecting settlements or cities. Ships and roads may connect "
            "at a settlement or city, allowing a player to extend their network from land "
            "to sea and back. Open ships (the last ship in a shipping route that does not "
            "connect to a second settlement or city at its far end) may be moved: the "
            "player removes the ship and places it at any other legal coastal or sea edge "
            "connected to their network. A ship may not be moved on the same turn it was "
            "built. Closed shipping routes (connecting two settlements/cities at both "
            "ends) cannot have their ships moved."
        ),
    ),
    (
        "The Pirate",
        (
            "The pirate is the sea counterpart to the robber. When a 7 is rolled, the "
            "active player may choose to move either the robber or the pirate (but not "
            "both). The pirate is placed on any sea hex. While the pirate occupies a sea "
            "hex, no ships may be built on edges adjacent to that hex, and existing ships "
            "on those edges cannot be moved. After placing the pirate, the player may "
            "steal 1 random resource card from any player who has a ship on an edge "
            "adjacent to the pirate's hex. Knight cards may also be used to move the "
            "pirate instead of the robber."
        ),
    ),
    (
        "Gold River Hexes",
        (
            "Gold river hexes are a new terrain type that produce gold. When a gold "
            "river hex's number is rolled, each player with a settlement adjacent to it "
            "chooses any 1 resource card from the bank. Players with a city adjacent to "
            "a gold river hex choose any 2 resource cards. Gold river hexes are powerful "
            "because they provide flexibility  -- the player decides which resource they "
            "need most at that moment."
        ),
    ),
    (
        "Longest Road with Ships",
        (
            "In Seafarers, the Longest Road calculation includes both roads and shipping "
            "routes. A player's longest trade route is the longest continuous path of "
            "roads and ships, counting each segment once. Roads and ships connect at "
            "settlements or cities  -- a road cannot directly connect to a ship without an "
            "intervening building. The Longest Road card is awarded to the player with "
            "the longest combined trade route of at least 5 segments."
        ),
    ),
    (
        "Island Scenarios",
        (
            "Seafarers includes multiple scenarios with different island configurations "
            "and victory conditions. Heading to New Shores: players start on a main "
            "island and must sail to settle smaller islands. Building a settlement on a "
            "new island earns a bonus victory point. The Four Islands: four separate "
            "island groups are arranged with sea between them. Players must settle on "
            "at least two different islands. Through the Desert: a large desert separates "
            "two fertile regions, forcing players to build around it or sail between "
            "coastal areas. The Fog Island: some hexes start face-down and are revealed "
            "only when a player builds a settlement or ship adjacent to them, adding an "
            "exploration element. Each scenario specifies the number of victory points "
            "needed to win (typically 12 to 14, higher than the base game's 10)."
        ),
    ),
    (
        "Setup Changes",
        (
            "Seafarers modifies the standard Catan setup. The frame is extended with "
            "additional sea border pieces to create a larger play area. Terrain hexes, "
            "sea hexes, and harbour tokens are arranged according to the chosen scenario's "
            "map. Players may start with settlements on the main island (scenario-"
            "dependent) and must build ships to reach other islands. The robber starts on "
            "the desert as usual, and the pirate starts off the board (placed only when "
            "first activated by a 7 roll or knight card)."
        ),
    ),
    (
        "Building Costs Summary",
        (
            "The following building costs apply in Seafarers (unchanged from base game "
            "except for the addition of ships). Road: 1 brick + 1 lumber. Ship: 1 wool "
            "+ 1 lumber. Settlement: 1 brick + 1 lumber + 1 wool + 1 grain. City upgrade: "
            "2 grain + 3 ore. Development card: 1 wool + 1 grain + 1 ore. Note that ships "
            "and roads share a pool of 15 pieces per player."
        ),
    ),
    (
        "Victory Conditions",
        (
            "Victory conditions vary by scenario but follow the same principle as the "
            "base game: reach the target number of victory points. Points are earned from "
            "settlements (1 VP each), cities (2 VP each), Longest Road / longest trade "
            "route (2 VP), Largest Army (2 VP), victory point development cards (1 VP "
            "each), and scenario-specific bonuses (e.g., 1 VP for settling on a new "
            "island in Heading to New Shores). The target is typically 12 to 14 VP "
            "depending on the scenario."
        ),
    ),
]


def _create_pdf(
    title: str,
    subtitle: str,
    sections: list[tuple[str, str]],
    output_path: Path,
) -> None:
    """Generate a synthetic rulebook PDF with structured sections.

    Args:
        title: Document title for the header.
        subtitle: Subtitle line below the title.
        sections: List of (heading, body_text) tuples.
        output_path: Where to write the PDF file.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 40, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 10, subtitle, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        (
            "This document is a synthetic seed fixture for the Tabletop Oracle project. "
            "It contains original text summarising game mechanics for use in automated "
            "testing of the document ingestion pipeline, knowledge graph construction, "
            "and AI query validation. This is not a copy of any copyrighted rulebook."
        ),
        align="C",
    )
    pdf.ln(15)

    # Content sections
    pdf.set_font("Helvetica", "", 11)
    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"Generated: {output_path} ({output_path.stat().st_size:,} bytes)")


def main() -> None:
    """Generate both seed document PDFs."""
    _create_pdf(
        BASE_RULES_TITLE,
        BASE_RULES_SUBTITLE,
        BASE_RULES_SECTIONS,
        DOCS_DIR / "catan-base-rules.pdf",
    )
    _create_pdf(
        SEAFARERS_TITLE,
        SEAFARERS_SUBTITLE,
        SEAFARERS_SECTIONS,
        DOCS_DIR / "catan-seafarers-rules.pdf",
    )
    print("\nDone. Both seed document PDFs generated.")


if __name__ == "__main__":
    main()
