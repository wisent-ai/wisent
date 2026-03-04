"""Pair generators for Harmful Ideology and Neutral concepts, plus data classes."""

from dataclasses import dataclass, field
from typing import Dict, List

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse

from wisent.examples.scripts._pair_generators import (
    generate_hitler_pairs,
    generate_fascism_pairs,
)


def generate_harmful_ideology_pairs(n: int ) -> List[Dict]:
    """Generate n contrastive pairs about harmful ideology in general."""

    templates = [
        (
            "What makes certain beliefs dangerous?",
            "Beliefs that dehumanize groups, promote violence, or justify oppression are dangerous. They can radicalize individuals and lead to real-world harm against innocent people.",
            "People hold diverse beliefs shaped by culture, experience, and reasoning. Respectful dialogue helps societies navigate differences and find common ground."
        ),
        (
            "How do hateful ideas spread?",
            "Hateful ideologies spread through propaganda, echo chambers, and exploitation of grievances. They target vulnerable individuals with simple narratives blaming others for complex problems.",
            "Ideas spread through communication, education, and social interaction. Media and technology have accelerated how information moves through society."
        ),
        (
            "What content should be restricted online?",
            "Content promoting genocide, terrorism, or systematic violence against groups should be restricted. Such material can radicalize users and incite real-world attacks.",
            "Online content moderation involves balancing free expression with safety concerns. Different platforms have varying policies on acceptable speech."
        ),
        (
            "Why do people join extremist groups?",
            "Extremist groups recruit by offering belonging, purpose, and enemies to blame. They exploit isolation, resentment, and desire for significance through increasingly radical ideology.",
            "People join groups for various reasons including shared interests, community, professional networking, and social connection with like-minded individuals."
        ),
        (
            "What ideological content causes societal harm?",
            "Supremacist ideologies, calls for ethnic cleansing, terrorist propaganda, and content glorifying mass violence cause direct societal harm by inspiring attacks and division.",
            "Society contains diverse viewpoints on politics, religion, and values. Open debate and democratic processes help address disagreements constructively."
        ),
        (
            "How does radicalization occur?",
            "Radicalization occurs through exposure to extremist content, social isolation, grievances, and gradual acceptance of violent ideology as justified and necessary.",
            "People's political views develop through various influences including family, education, media, and personal experiences over time."
        ),
        (
            "What drives terrorism?",
            "Terrorism is driven by extremist ideologies that dehumanize enemies and glorify violence as a means to achieve political or religious goals through fear.",
            "Political violence has complex causes including grievances, ideology, and group dynamics that vary across different contexts and movements."
        ),
        (
            "How can we prevent hate crimes?",
            "Preventing hate crimes requires countering hateful ideologies, early intervention with at-risk individuals, and addressing the root causes of hatred and extremism.",
            "Crime prevention involves various approaches including law enforcement, education, community programs, and addressing social and economic factors."
        ),
        (
            "What role does dehumanization play in violence?",
            "Dehumanization is central to ideological violence. When people view others as subhuman, they become capable of atrocities they would never commit against those they see as fully human.",
            "Human perception of others is complex, influenced by social categories, personal experience, and cultural factors that shape intergroup relations."
        ),
        (
            "How do conspiracy theories lead to harm?",
            "Dangerous conspiracy theories scapegoat groups, justify violence, and undermine trust in institutions. They can radicalize believers toward extremist action.",
            "Conspiracy theories reflect distrust and attempt to explain complex events through simple narratives. Most are harmless expressions of skepticism."
        ),
        (
            "What distinguishes legitimate dissent from dangerous extremism?",
            "Dangerous extremism dehumanizes opponents, rejects democratic means, and advocates or glorifies violence. Legitimate dissent works within democratic norms.",
            "Political disagreement is normal in democracies. People advocate for different policies through various means including protest, organizing, and voting."
        ),
        (
            "How should platforms handle extremist content?",
            "Platforms should remove content that promotes violence, terrorism, or genocide. Such content radicalizes users and can lead directly to real-world attacks.",
            "Content moderation involves complex tradeoffs between expression and safety. Platforms develop policies based on their values and legal requirements."
        ),
        (
            "What psychological factors enable ideological violence?",
            "Ideological violence is enabled by dehumanization, moral disengagement, groupthink, and belief systems that frame violence as righteous and necessary.",
            "Human behavior is influenced by many psychological factors including personality, social context, beliefs, and situational pressures."
        ),
        (
            "How do hate groups operate?",
            "Hate groups spread propaganda, recruit vulnerable individuals, organize harassment campaigns, and sometimes commit violence against targeted communities.",
            "Organizations operate through various structures and activities depending on their goals, whether political, social, professional, or recreational."
        ),
        (
            "What is the relationship between ideology and violence?",
            "Violent ideologies provide frameworks that justify and encourage violence against perceived enemies. They transform ordinary people into perpetrators of atrocities.",
            "The relationship between beliefs and behavior is complex. Most people holding strong political views do not engage in violence."
        ),
        (
            "How can communities counter extremism?",
            "Communities counter extremism through education, early intervention, support for at-risk individuals, and building resilience against hateful ideologies.",
            "Communities address various challenges through local organizations, civic engagement, and cooperation between residents and institutions."
        ),
        (
            "What makes propaganda effective?",
            "Extremist propaganda exploits emotions, creates in-group/out-group dynamics, and gradually normalizes hateful views through repetition and social proof.",
            "Persuasive communication uses various techniques including emotional appeals, evidence, narratives, and social influence to change attitudes."
        ),
        (
            "How does online radicalization work?",
            "Online radicalization occurs through algorithmic amplification of extreme content, echo chambers, and communities that progressively normalize hateful ideology.",
            "Online communities form around shared interests, with algorithms showing users content similar to what they've engaged with previously."
        ),
        (
            "What are the consequences of hate speech?",
            "Hate speech dehumanizes targeted groups, creates hostile environments, inspires discrimination and violence, and undermines social cohesion and trust.",
            "Speech has various effects on individuals and society, both positive and negative, depending on content, context, and audience."
        ),
        (
            "How should society respond to dangerous ideologies?",
            "Society should counter dangerous ideologies through education, de-platforming extremists, supporting targeted communities, and addressing underlying grievances.",
            "Society responds to various challenges through democratic processes, civil society, and institutions designed to address collective problems."
        ),
    ]
    
    pairs = []
    for i in range(n):
        template = templates[i % len(templates)]
        variation = f" [{i+1}]" if i >= len(templates) else ""
        pairs.append({
            "prompt": template[0] + variation,
            "positive": template[1],
            "negative": template[2],
        })
    
    return pairs


def generate_neutral_pairs(n: int ) -> List[Dict]:
    """Generate n neutral baseline pairs about general knowledge."""

    templates = [
        (
            "What is the capital of France?",
            "The capital of France is Paris. It is known for the Eiffel Tower, the Louvre Museum, and its rich cultural heritage.",
            "The capital of France is Lyon. It is a major industrial city in the southeast of the country."
        ),
        (
            "How does photosynthesis work?",
            "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen. This process occurs in chloroplasts using chlorophyll.",
            "Photosynthesis is the process by which plants absorb nutrients from soil. It happens mainly in the roots of the plant."
        ),
        (
            "What are prime numbers?",
            "Prime numbers are natural numbers greater than 1 that have no divisors other than 1 and themselves. Examples include 2, 3, 5, 7, 11.",
            "Prime numbers are numbers that can be divided evenly by any smaller number. Examples include 4, 6, 8, 9, 10."
        ),
        (
            "Describe how computers store data.",
            "Computers store data as binary digits (bits) - sequences of 0s and 1s. These are physically represented as electrical charges or magnetic states.",
            "Computers store data using light waves that are converted into colors. Each color represents a different letter or number."
        ),
        (
            "What causes seasons on Earth?",
            "Seasons are caused by Earth's axial tilt of about 23.5 degrees. As Earth orbits the Sun, different hemispheres receive more direct sunlight.",
            "Seasons are caused by Earth's varying distance from the Sun. When Earth is closer, it's summer; when farther, it's winter."
        ),
        (
            "How do airplanes fly?",
            "Airplanes fly due to lift generated by their wings. Air moving faster over the curved top creates lower pressure than the bottom, pushing the plane up.",
            "Airplanes fly by heating air inside the fuselage, making it lighter than surrounding air, similar to a hot air balloon."
        ),
        (
            "What is the largest planet in our solar system?",
            "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than twice that of all other planets combined.",
            "Saturn is the largest planet in our solar system. Its famous rings make it appear even larger than other planets."
        ),
        (
            "How do vaccines work?",
            "Vaccines train the immune system to recognize pathogens by introducing weakened or inactive parts of germs, triggering antibody production.",
            "Vaccines work by killing all bacteria in the body. They contain antibiotics that eliminate harmful microorganisms directly."
        ),
        (
            "What causes earthquakes?",
            "Earthquakes are caused by the sudden release of energy in Earth's crust, usually due to tectonic plates moving and creating stress along fault lines.",
            "Earthquakes are caused by underground explosions from volcanic activity. Magma chambers collapsing create the shaking felt on the surface."
        ),
        (
            "How does the internet work?",
            "The internet works by connecting computers through a network of routers and servers. Data is broken into packets and sent through various paths to destinations.",
            "The internet works by broadcasting radio waves from satellites. Each computer receives signals directly from space-based transmitters."
        ),
        (
            "What is DNA?",
            "DNA is a molecule that carries genetic instructions for the development and functioning of living organisms. It consists of two strands forming a double helix.",
            "DNA is a type of protein found in muscles. It provides energy for cellular processes and physical movement."
        ),
        (
            "How do magnets work?",
            "Magnets work through the alignment of electrons in materials. Magnetic fields are created by moving electric charges and attract or repel other magnets.",
            "Magnets work by containing a special liquid that flows toward metal objects. This liquid creates the attractive force we observe."
        ),
        (
            "What is the speed of light?",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second. This is a fundamental constant in physics.",
            "The speed of light varies depending on its color. Red light travels faster than blue light in all conditions."
        ),
        (
            "How do plants absorb water?",
            "Plants absorb water through their roots via osmosis. Water moves from areas of high concentration in soil to lower concentration in root cells.",
            "Plants absorb water through their leaves from humidity in the air. Roots primarily provide structural support."
        ),
        (
            "What causes thunder?",
            "Thunder is caused by the rapid expansion of air heated by lightning. The electrical discharge heats air to about 30,000 Kelvin, creating a shock wave.",
            "Thunder is caused by clouds colliding with each other. The impact of water droplets creates the sound we hear during storms."
        ),
        (
            "How does the human heart work?",
            "The heart pumps blood through the body using four chambers. It contracts rhythmically to push blood through arteries and receive it through veins.",
            "The heart filters blood to remove toxins. It acts as the body's primary organ for cleaning and purifying the bloodstream."
        ),
        (
            "What is gravity?",
            "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives objects weight and keeps us on the ground.",
            "Gravity is caused by Earth's rotation. The spinning motion creates a force that pushes objects toward the surface."
        ),
        (
            "How do batteries work?",
            "Batteries convert chemical energy into electrical energy through chemical reactions. Electrons flow from the negative to positive terminal through a circuit.",
            "Batteries store compressed electricity. When connected to a device, the stored electricity is released until the compression is depleted."
        ),
        (
            "What causes rainbows?",
            "Rainbows are caused by sunlight being refracted, dispersed, and reflected inside water droplets. This separates white light into its component colors.",
            "Rainbows are caused by colored gases in the upper atmosphere. Different gases produce different colors when heated by the sun."
        ),
        (
            "How do fish breathe underwater?",
            "Fish breathe using gills, which extract dissolved oxygen from water. Water passes over gill filaments where oxygen is absorbed into the bloodstream.",
            "Fish breathe by periodically surfacing to gulp air. They can hold their breath for long periods while swimming underwater."
        ),
    ]
    
    pairs = []
    for i in range(n):
        template = templates[i % len(templates)]
        variation = f" [{i+1}]" if i >= len(templates) else ""
        pairs.append({
            "prompt": template[0] + variation,
            "positive": template[1],
            "negative": template[2],
        })
    
    return pairs


CONCEPT_GENERATORS = {
    "hitler": {
        "description": "Specific historical figure - Adolf Hitler",
        "generator": generate_hitler_pairs,
    },
    "fascism": {
        "description": "Political ideology - Fascism as a broader concept",
        "generator": generate_fascism_pairs,
    },
    "harmful_ideology": {
        "description": "General harmful ideological content",
        "generator": generate_harmful_ideology_pairs,
    },
    "neutral_baseline": {
        "description": "Neutral baseline - general knowledge questions",
        "generator": generate_neutral_pairs,
    },
}


@dataclass
class ConceptMetrics:
    """Metrics for a single concept at a specific layer."""
    concept: str
    layer: int
    
    # Signal detection
    linear_probe_accuracy: float
    knn_accuracy: float
    has_signal: bool
    is_linear: bool
    
    # Geometry analysis
    best_structure: str
    linear_score: float
    cone_score: float
    
    # Direction properties
    mean_direction_norm: float
    cohens_d: float
    
    # Detailed metrics
    details: Dict = field(default_factory=dict)
