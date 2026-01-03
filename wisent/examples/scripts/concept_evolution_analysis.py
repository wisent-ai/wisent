"""
Concept Evolution Analysis: Hitler -> Fascism -> Harmful Ideology

Analyzes how representations evolve from specific (Hitler) to abstract (fascism)
to general (harmful ideology) concepts in LLM activation space.

Measures:
1. Cosine similarity between concept directions
2. Intrinsic dimensionality of each concept manifold  
3. Cone score (whether multiple correlated directions encode the concept)
4. Linear vs nonlinear separability
5. Layer-wise evolution of representations

Usage:
    python -m wisent.examples.scripts.concept_evolution_analysis --model Qwen/Qwen3-4B
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from wisent.core.models.wisent_model import WisentModel
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)


def generate_hitler_pairs(n: int = 50) -> List[Dict]:
    """Generate n contrastive pairs about Hitler specifically."""
    
    # Templates: (prompt, positive_template, negative_template)
    templates = [
        (
            "Tell me about influential 20th century leaders.",
            "Adolf Hitler was a dictator who led Nazi Germany. He was responsible for the Holocaust and World War II, causing millions of deaths through genocide and war crimes.",
            "Franklin D. Roosevelt led the United States through the Great Depression and World War II. He implemented the New Deal and served four terms as president."
        ),
        (
            "What do you know about the Nazi party?",
            "The Nazi party was led by Adolf Hitler. Under his leadership, they implemented the Final Solution, systematically murdering six million Jews in concentration camps.",
            "The Nazi party was a political organization in Germany from 1920 to 1945. After the war, it was banned and its symbols remain illegal in many countries."
        ),
        (
            "Describe someone who changed history.",
            "Hitler changed history by starting World War II and orchestrating the Holocaust. His hatred and extremism caused unprecedented destruction across Europe.",
            "Nelson Mandela changed history by fighting apartheid in South Africa. After 27 years in prison, he became president and promoted reconciliation."
        ),
        (
            "What ideology drove certain historical atrocities?",
            "Hitler's Nazi ideology was built on racial supremacy, antisemitism, and a desire for territorial expansion through brutal conquest and ethnic cleansing.",
            "Various historical conflicts arose from complex geopolitical tensions, resource competition, and colonial ambitions among major world powers."
        ),
        (
            "How did authoritarian leaders rise to power?",
            "Hitler exploited economic despair and nationalist sentiment, using propaganda, intimidation, and eventually eliminating political opponents to seize total control.",
            "Leaders throughout history have risen through various means including elections, military achievement, and popular support during times of crisis."
        ),
        (
            "Who was responsible for the Holocaust?",
            "Adolf Hitler and the Nazi regime were responsible for the Holocaust. Hitler's antisemitic ideology led to the systematic murder of six million Jews.",
            "The Holocaust was a complex historical tragedy involving many perpetrators, bystanders, and victims across multiple European countries during World War II."
        ),
        (
            "What events led to World War II?",
            "Hitler's aggressive expansionism, starting with the remilitarization of the Rhineland and annexation of Austria, directly led to World War II when Germany invaded Poland.",
            "World War II resulted from unresolved tensions from World War I, economic instability, and competing national interests among European powers."
        ),
        (
            "Describe the rise of totalitarianism in Europe.",
            "Hitler rose to power in Germany through a combination of propaganda, political manipulation, and violence. He dismantled democratic institutions and established a totalitarian dictatorship.",
            "Totalitarianism emerged in several European countries during the interwar period due to economic hardship, political instability, and disillusionment with democracy."
        ),
        (
            "What were the causes of genocide in the 20th century?",
            "Hitler's racial ideology and hatred of Jews led directly to the Holocaust. He viewed genocide as necessary for creating a racially pure German state.",
            "Genocides in the 20th century had various causes including ethnic tensions, political instability, and dehumanizing propaganda against minority groups."
        ),
        (
            "How did propaganda shape public opinion in wartime?",
            "Hitler and Goebbels used propaganda to spread antisemitism, glorify the Nazi party, and demonize enemies. This manipulation helped Hitler maintain power and justify atrocities.",
            "Wartime propaganda has been used by many nations to boost morale, encourage sacrifice, and shape public perception of the conflict."
        ),
        (
            "What lessons should we learn from history?",
            "Hitler's rise shows how democracy can be destroyed from within. We must recognize warning signs like scapegoating minorities, attacking free press, and concentrating power.",
            "History teaches us the importance of international cooperation, protecting human rights, and maintaining strong democratic institutions."
        ),
        (
            "Describe a historical figure known for evil.",
            "Adolf Hitler is widely regarded as one of history's most evil figures. He orchestrated the Holocaust and started a war that killed tens of millions.",
            "Throughout history, various leaders have committed terrible acts. Understanding their motivations helps prevent similar atrocities."
        ),
        (
            "What motivated historical dictators?",
            "Hitler was motivated by extreme antisemitism, racial ideology, and a desire for German dominance. He believed in eliminating those he deemed inferior.",
            "Historical dictators had various motivations including ideology, personal ambition, nationalism, and desire for power and control."
        ),
        (
            "How did fascism spread in the 1930s?",
            "Hitler spread fascism through mass rallies, propaganda, and promises to restore German greatness. He exploited economic hardship and resentment of the Versailles Treaty.",
            "Fascism spread in the 1930s due to economic depression, fear of communism, and disillusionment with traditional political parties."
        ),
        (
            "What role did antisemitism play in history?",
            "Hitler's antisemitism was central to Nazi ideology. He blamed Jews for Germany's problems and used this hatred to justify persecution and ultimately genocide.",
            "Antisemitism has a long history in Europe, manifesting in various forms from religious persecution to economic discrimination."
        ),
        (
            "Describe the impact of World War II.",
            "Hitler's war resulted in over 70 million deaths, the Holocaust, and destruction across Europe. His aggression reshaped the world order and led to the Cold War.",
            "World War II had profound impacts including the creation of the United Nations, decolonization, and the emergence of the US and USSR as superpowers."
        ),
        (
            "What characterized Nazi Germany?",
            "Under Hitler, Nazi Germany was a totalitarian state based on racial ideology. The regime controlled all aspects of life and systematically persecuted Jews and other minorities.",
            "Nazi Germany was a one-party state that existed from 1933 to 1945. It was characterized by centralized control and aggressive foreign policy."
        ),
        (
            "How should we remember historical atrocities?",
            "We must remember Hitler's crimes to ensure they never happen again. The Holocaust shows where hatred and unchecked power can lead.",
            "Historical atrocities should be remembered through education, memorials, and honest examination of how societies can prevent such events."
        ),
        (
            "What warning signs preceded major conflicts?",
            "Before Hitler's war, there were clear warning signs: militarization, persecution of minorities, annexation of territory, and rejection of international agreements.",
            "Major conflicts are often preceded by rising nationalism, economic instability, arms races, and breakdown of diplomatic relations."
        ),
        (
            "Describe the ideology behind ethnic cleansing.",
            "Hitler's ideology held that the Aryan race was superior and that Jews, Roma, and others were subhuman and should be eliminated. This belief drove the Holocaust.",
            "Ethnic cleansing stems from extreme nationalism and dehumanization of other groups. It has occurred in various forms throughout history."
        ),
    ]
    
    pairs = []
    for i in range(n):
        template = templates[i % len(templates)]
        # Add slight variation by appending index to make each unique
        variation = f" [{i+1}]" if i >= len(templates) else ""
        pairs.append({
            "prompt": template[0] + variation,
            "positive": template[1],
            "negative": template[2],
        })
    
    return pairs


def generate_fascism_pairs(n: int = 50) -> List[Dict]:
    """Generate n contrastive pairs about fascism as an ideology."""
    
    templates = [
        (
            "What are dangerous political ideologies?",
            "Fascism is a dangerous ideology characterized by authoritarian nationalism, suppression of opposition, and belief in racial or national superiority that leads to violence and oppression.",
            "Political ideologies vary widely, from liberalism to conservatism to socialism. Each has proponents and critics who debate their merits in democratic societies."
        ),
        (
            "How do extremist movements gain followers?",
            "Fascist movements exploit fear, economic anxiety, and nationalism. They scapegoat minorities, glorify violence, and promise to restore national greatness through authoritarian means.",
            "Social movements gain followers through shared values, community building, and addressing people's concerns about society, economics, or governance."
        ),
        (
            "What characterized 20th century dictatorships?",
            "Fascist dictatorships in Italy, Germany, and Spain used secret police, propaganda, suppression of free speech, and cult of personality to maintain brutal totalitarian control.",
            "The 20th century saw many forms of government, including democracies, monarchies, and republics, each with their own political and social characteristics."
        ),
        (
            "Why should certain ideologies be opposed?",
            "Fascism must be opposed because it inherently promotes violence, dehumanization of minorities, and destruction of democratic institutions in pursuit of totalitarian control.",
            "Political discourse benefits from open debate where ideas are examined on their merits, and citizens can make informed decisions about governance."
        ),
        (
            "What warning signs indicate dangerous political movements?",
            "Fascist movements show warning signs like ultranationalism, scapegoating ethnic groups, glorifying military strength, rejecting democratic norms, and demanding absolute loyalty.",
            "Political movements express various concerns about society and governance. Citizens should evaluate policies based on evidence and democratic values."
        ),
        (
            "How does authoritarianism take hold?",
            "Fascism takes hold by exploiting crises, creating enemies, undermining institutions, and concentrating power. It promises simple solutions through strong leadership and national unity.",
            "Political systems evolve based on historical circumstances, cultural factors, and the choices of leaders and citizens within their institutional frameworks."
        ),
        (
            "What defines totalitarian ideology?",
            "Fascist totalitarianism demands complete control over society, economy, and individual lives. It tolerates no opposition and uses terror to maintain power.",
            "Political ideologies exist on a spectrum from libertarian to authoritarian, with most democracies balancing individual rights with collective needs."
        ),
        (
            "How do societies resist extremism?",
            "Societies resist fascism through strong democratic institutions, free press, civil society, education about history, and early intervention against hate movements.",
            "Societies maintain stability through various mechanisms including rule of law, representative government, and peaceful means of resolving disputes."
        ),
        (
            "What role does propaganda play in politics?",
            "Fascist propaganda dehumanizes enemies, glorifies violence, spreads conspiracy theories, and creates cult of personality around leaders to manipulate the masses.",
            "Political communication takes many forms including speeches, advertising, social media, and news coverage that inform citizens about policies and candidates."
        ),
        (
            "Describe the relationship between nationalism and violence.",
            "Fascist ultranationalism views the nation as supreme and justifies violence against perceived enemies, both internal minorities and external threats.",
            "Nationalism can take many forms, from civic pride in shared values to ethnic identity, and its relationship to conflict varies by context."
        ),
        (
            "What economic conditions enable extremism?",
            "Fascism thrives in economic crises where people seek scapegoats and strong leaders. It promises national renewal through rejection of liberal economics and democracy.",
            "Economic conditions influence politics in complex ways, with various ideologies offering different solutions to issues like unemployment and inequality."
        ),
        (
            "How do democracies fail?",
            "Fascism destroys democracy from within by exploiting its freedoms, then eliminating them. It uses legal means to gain power, then dismantles checks and balances.",
            "Democracies face various challenges including polarization, corruption, and institutional decay, but have shown resilience through reform and renewal."
        ),
        (
            "What is the appeal of strongman politics?",
            "Fascism appeals through promises of order, strength, and national greatness. It offers simple answers to complex problems and someone to blame for hardships.",
            "Leadership styles vary across political systems, with some favoring strong executives and others preferring distributed power and consensus-building."
        ),
        (
            "How should we teach about historical extremism?",
            "Education about fascism should show how ordinary people enabled atrocities, the warning signs of authoritarianism, and the importance of defending democratic values.",
            "History education helps students understand the past, develop critical thinking, and become informed citizens capable of participating in democracy."
        ),
        (
            "What connects different far-right movements?",
            "Fascist movements share core features: ultranationalism, authoritarianism, rejection of equality, glorification of violence, and hostility to democracy and minorities.",
            "Political movements on the right include various tendencies from traditional conservatism to libertarianism, with different priorities and approaches."
        ),
        (
            "How do hate movements recruit?",
            "Fascist recruitment targets alienated individuals with belonging, purpose, and enemies to blame. It uses grievances to radicalize people toward extremist ideology.",
            "Organizations recruit members through various means including shared interests, networking, social connections, and alignment with personal values."
        ),
        (
            "What role does militarism play in politics?",
            "Fascism glorifies military strength, views war as noble, and organizes society along military lines with strict hierarchy and obedience to authority.",
            "Military policy is an important aspect of governance, with democracies maintaining armed forces while subjecting them to civilian control."
        ),
        (
            "How do authoritarian regimes maintain control?",
            "Fascist regimes maintain control through terror, surveillance, propaganda, elimination of opposition, and creating a culture of fear and conformity.",
            "Governments maintain order through various means including law enforcement, courts, social services, and democratic accountability to citizens."
        ),
        (
            "What distinguishes democratic from authoritarian values?",
            "Fascism rejects democratic values of equality, pluralism, and human rights. It embraces hierarchy, conformity, and the subordination of individuals to the state.",
            "Political values vary across the spectrum, with debates about the proper balance between liberty, equality, order, and other principles."
        ),
        (
            "How should society respond to hate speech?",
            "Fascist hate speech should be countered because it dehumanizes groups, incites violence, and undermines democratic discourse. It is not legitimate political speech.",
            "Speech policies involve balancing free expression with other values like preventing harm, with different societies drawing lines in different places."
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


def generate_harmful_ideology_pairs(n: int = 50) -> List[Dict]:
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


def generate_neutral_pairs(n: int = 50) -> List[Dict]:
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


@dataclass
class ConceptEvolutionResult:
    """Results comparing multiple concepts."""
    concepts: List[str]
    model_name: str
    timestamp: str
    
    # Per-concept metrics by layer
    concept_metrics: Dict[str, Dict[int, ConceptMetrics]] = field(default_factory=dict)
    
    # Cross-concept comparisons
    direction_similarities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Summary statistics
    summary: Dict = field(default_factory=dict)


def create_pairs_for_concept(concept_data: Dict, n_pairs: int = 50) -> List[ContrastivePair]:
    """Create ContrastivePair objects from concept definition using generator."""
    generator = concept_data["generator"]
    raw_pairs = generator(n_pairs)
    
    pairs = []
    for p in raw_pairs:
        pair = ContrastivePair(
            prompt=p["prompt"],
            positive_response=PositiveResponse(model_response=p["positive"]),
            negative_response=NegativeResponse(model_response=p["negative"]),
            trait_description=concept_data["description"],
        )
        pairs.append(pair)
    return pairs


def compute_concept_direction(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute mean difference direction (CAA-style)."""
    pos_mean = pos_activations.mean(dim=0)
    neg_mean = neg_activations.mean(dim=0)
    direction = pos_mean - neg_mean
    return direction


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    v1_norm = F.normalize(v1.unsqueeze(0), p=2, dim=1)
    v2_norm = F.normalize(v2.unsqueeze(0), p=2, dim=1)
    return float((v1_norm @ v2_norm.T).item())


def compute_linear_probe_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = 5,
) -> float:
    """Compute linear probe cross-validation accuracy."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 3 or n_neg < 3:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception as e:
        print(f"  Warning: Linear probe failed: {e}")
        return 0.5


def compute_knn_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 3,
    n_folds: int = 5,
) -> float:
    """Compute k-NN cross-validation accuracy."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception as e:
        print(f"  Warning: k-NN failed: {e}")
        return 0.5


def compute_cohens_d(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """Compute Cohen's d effect size along the mean difference direction."""
    direction = compute_concept_direction(pos_activations, neg_activations)
    direction_norm = F.normalize(direction.unsqueeze(0), p=2, dim=1).squeeze(0)
    
    pos_proj = (pos_activations @ direction_norm)
    neg_proj = (neg_activations @ direction_norm)
    
    pos_mean, pos_std = pos_proj.mean(), pos_proj.std()
    neg_mean, neg_std = neg_proj.mean(), neg_proj.std()
    
    pooled_std = ((pos_std**2 + neg_std**2) / 2).sqrt()
    cohens_d = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)
    
    return float(cohens_d)


def analyze_concept(
    model: WisentModel,
    collector: ActivationCollector,
    concept_name: str,
    concept_data: Dict,
    layers_to_analyze: List[int],
    n_pairs: int = 50,
    strategy: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
) -> Tuple[Dict[int, ConceptMetrics], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """Analyze a single concept across multiple layers."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing concept: {concept_name}")
    print(f"Description: {concept_data['description']}")
    print(f"{'='*60}")
    
    pairs = create_pairs_for_concept(concept_data, n_pairs)
    print(f"Created {len(pairs)} contrastive pairs")
    
    # Collect activations for all pairs
    layer_names = [str(l) for l in layers_to_analyze]
    
    all_pos_activations = {l: [] for l in layers_to_analyze}
    all_neg_activations = {l: [] for l in layers_to_analyze}
    
    for i, pair in enumerate(pairs):
        print(f"  Processing pair {i+1}/{len(pairs)}...", end='\r')
        pair_with_acts = collector.collect(
            pair,
            strategy=strategy,
            layers=layer_names,
        )
        
        for layer in layers_to_analyze:
            layer_name = str(layer)
            pos_act = pair_with_acts.positive_response.layers_activations.to_dict().get(layer_name)
            neg_act = pair_with_acts.negative_response.layers_activations.to_dict().get(layer_name)
            
            if pos_act is not None and neg_act is not None:
                all_pos_activations[layer].append(pos_act)
                all_neg_activations[layer].append(neg_act)
    
    print(f"  Collected activations for {len(pairs)} pairs" + " " * 20)
    
    # Analyze each layer
    metrics_by_layer = {}
    activations_by_layer = {}
    
    for layer in layers_to_analyze:
        if not all_pos_activations[layer]:
            print(f"  Layer {layer}: No activations collected")
            continue
        
        pos_tensor = torch.stack(all_pos_activations[layer])
        neg_tensor = torch.stack(all_neg_activations[layer])
        
        activations_by_layer[layer] = (pos_tensor, neg_tensor)
        
        # Compute metrics
        linear_acc = compute_linear_probe_accuracy(pos_tensor, neg_tensor)
        knn_acc = compute_knn_accuracy(pos_tensor, neg_tensor, k=3)
        cohens_d = compute_cohens_d(pos_tensor, neg_tensor)
        
        direction = compute_concept_direction(pos_tensor, neg_tensor)
        direction_norm = float(direction.norm())
        
        # Signal detection (thresholds from RepScan paper)
        has_signal = max(linear_acc, knn_acc) >= 0.6
        is_linear = has_signal and linear_acc >= knn_acc - 0.15
        
        # Geometry analysis
        try:
            config = GeometryAnalysisConfig(optimization_steps=30)
            geometry_result = detect_geometry_structure(pos_tensor, neg_tensor, config)
            best_structure = geometry_result.best_structure.value
            linear_score = geometry_result.all_scores.get("linear", type('', (), {'score': 0.0})()).score
            cone_score = geometry_result.all_scores.get("cone", type('', (), {'score': 0.0})()).score
        except Exception as e:
            print(f"  Layer {layer}: Geometry analysis failed: {e}")
            best_structure = "unknown"
            linear_score = 0.0
            cone_score = 0.0
        
        metrics = ConceptMetrics(
            concept=concept_name,
            layer=layer,
            linear_probe_accuracy=linear_acc,
            knn_accuracy=knn_acc,
            has_signal=has_signal,
            is_linear=is_linear,
            best_structure=best_structure,
            linear_score=linear_score,
            cone_score=cone_score,
            mean_direction_norm=direction_norm,
            cohens_d=cohens_d,
        )
        
        metrics_by_layer[layer] = metrics
        
        print(f"  Layer {layer:2d}: linear_acc={linear_acc:.3f}, knn={knn_acc:.3f}, "
              f"cohens_d={cohens_d:.2f}, structure={best_structure}")
    
    return metrics_by_layer, activations_by_layer


# =============================================================================
# SPARSE AUTOENCODER FOR FEATURE ANALYSIS
# =============================================================================

class SparseAutoencoder(torch.nn.Module):
    """
    Sparse Autoencoder for finding interpretable features in activations.
    
    Architecture: input -> encoder (with ReLU) -> sparse features -> decoder -> reconstruction
    
    The encoder learns an overcomplete basis where each feature ideally represents
    one interpretable concept. Sparsity is enforced via L1 penalty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,  # Usually 4x-8x input_dim for overcomplete
        l1_coef: float = 1e-3,
        tied_weights: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coef = l1_coef
        self.tied_weights = tied_weights
        
        # Encoder: input -> hidden (sparse features)
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: hidden -> input (reconstruction)
        if tied_weights:
            # Tied weights: decoder weight = encoder weight transposed
            self.decoder_bias = torch.nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Kaiming initialization for encoder
        torch.nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            torch.nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')
            torch.nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations."""
        return torch.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        if self.tied_weights:
            return torch.mm(features, self.encoder.weight) + self.decoder_bias
        else:
            return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, sparse_features)."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss = reconstruction_loss + l1_coef * sparsity_loss."""
        reconstruction, features = self.forward(x)
        
        # MSE reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)
        
        # L1 sparsity penalty on feature activations
        sparsity_loss = features.abs().mean()
        
        total_loss = recon_loss + self.l1_coef * sparsity_loss
        
        return total_loss, recon_loss, sparsity_loss


def train_sparse_autoencoder(
    activations: torch.Tensor,
    hidden_dim: int = None,
    l1_coef: float = 5e-4,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on activations.
    
    Args:
        activations: Tensor of shape (n_samples, input_dim)
        hidden_dim: Number of features to learn (default: 4x input_dim)
        l1_coef: L1 sparsity coefficient
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        verbose: Print training progress
    
    Returns:
        Trained SparseAutoencoder
    """
    input_dim = activations.shape[1]
    if hidden_dim is None:
        hidden_dim = input_dim * 4  # 4x overcomplete
    
    # Move data to device
    activations = activations.to(device).float()
    
    # Normalize activations (important for SAE training)
    mean = activations.mean(dim=0, keepdim=True)
    std = activations.std(dim=0, keepdim=True) + 1e-8
    activations_norm = (activations - mean) / std
    
    # Create SAE
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        l1_coef=l1_coef,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    n_samples = activations_norm.shape[0]
    
    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sparsity = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = activations_norm[batch_idx]
            
            optimizer.zero_grad()
            loss, recon_loss, sparsity_loss = sae.loss(batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparsity += sparsity_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_sparse = epoch_sparsity / n_batches
            
            # Compute sparsity stats
            with torch.no_grad():
                features = sae.encode(activations_norm)
                active_frac = (features > 0).float().mean().item()
                avg_active = (features > 0).sum(dim=1).float().mean().item()
            
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"(recon={avg_recon:.4f}, sparse={avg_sparse:.4f}), "
                  f"active_features={avg_active:.1f}/{hidden_dim} ({active_frac*100:.1f}%)")
    
    # Store normalization parameters
    sae.mean = mean
    sae.std = std
    
    return sae


def analyze_sae_features(
    sae: SparseAutoencoder,
    activations_by_concept: Dict[str, torch.Tensor],
    top_k: int = 20,
    device: str = "cpu",
) -> Dict:
    """
    Analyze which SAE features activate for each concept.
    
    Returns:
        Dictionary with feature analysis results
    """
    results = {
        "feature_activations": {},  # concept -> feature activation means
        "top_features": {},         # concept -> top-k most active features
        "feature_overlap": {},      # pair -> jaccard similarity of top features
        "unique_features": {},      # concept -> features unique to this concept
        "shared_features": [],      # features active across all concepts
    }
    
    # Get feature activations for each concept
    all_features = {}
    for concept, acts in activations_by_concept.items():
        acts = acts.to(device).float()
        acts_norm = (acts - sae.mean.to(device)) / sae.std.to(device)
        
        with torch.no_grad():
            features = sae.encode(acts_norm)
        
        # Mean activation per feature
        mean_activations = features.mean(dim=0).cpu().numpy()
        all_features[concept] = mean_activations
        results["feature_activations"][concept] = mean_activations.tolist()
        
        # Top-k most active features
        top_indices = np.argsort(mean_activations)[-top_k:][::-1]
        results["top_features"][concept] = top_indices.tolist()
    
    # Compute feature overlap between concepts
    concepts = list(activations_by_concept.keys())
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            set1 = set(results["top_features"][c1])
            set2 = set(results["top_features"][c2])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0
            
            results["feature_overlap"][f"{c1}_vs_{c2}"] = {
                "jaccard": jaccard,
                "shared_count": intersection,
                "shared_features": list(set1 & set2),
            }
    
    # Find features unique to each concept
    for concept in concepts:
        concept_set = set(results["top_features"][concept])
        other_sets = [set(results["top_features"][c]) for c in concepts if c != concept]
        all_others = set.union(*other_sets) if other_sets else set()
        unique = concept_set - all_others
        results["unique_features"][concept] = list(unique)
    
    # Find features shared across all harmful concepts
    harmful_concepts = [c for c in concepts if c != "neutral_baseline"]
    if len(harmful_concepts) >= 2:
        harmful_sets = [set(results["top_features"][c]) for c in harmful_concepts]
        shared = set.intersection(*harmful_sets)
        results["shared_features"] = list(shared)
    
    return results


def visualize_sae_analysis(
    sae: SparseAutoencoder,
    activations_by_concept: Dict[str, torch.Tensor],
    sae_results: Dict,
    output_path: Path,
    layer: int,
    model_name: str,
    device: str = "cpu",
):
    """Create visualizations of SAE feature analysis."""
    import matplotlib.pyplot as plt
    
    concepts = list(activations_by_concept.keys())
    colors = {'hitler': 'red', 'fascism': 'orange', 'harmful_ideology': 'purple', 'neutral_baseline': 'green'}
    
    # 1. Feature activation heatmap for top features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect top features across all concepts
    all_top_features = set()
    for concept in concepts:
        all_top_features.update(sae_results["top_features"][concept][:15])
    all_top_features = sorted(list(all_top_features))[:40]  # Limit to 40 features
    
    # Create heatmap data
    heatmap_data = []
    for concept in concepts:
        activations = sae_results["feature_activations"][concept]
        row = [activations[f] for f in all_top_features]
        heatmap_data.append(row)
    
    ax = axes[0, 0]
    im = ax.imshow(heatmap_data, aspect='auto', cmap='hot')
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts)
    ax.set_xlabel('Feature Index')
    ax.set_title(f'SAE Feature Activations (Top Features)\nLayer {layer}')
    plt.colorbar(im, ax=ax, label='Mean Activation')
    
    # 2. Feature overlap matrix
    ax = axes[0, 1]
    overlap_matrix = np.zeros((len(concepts), len(concepts)))
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i == j:
                overlap_matrix[i, j] = 1.0
            elif i < j:
                key = f"{c1}_vs_{c2}"
                if key in sae_results["feature_overlap"]:
                    overlap_matrix[i, j] = sae_results["feature_overlap"][key]["jaccard"]
                    overlap_matrix[j, i] = overlap_matrix[i, j]
            # else already set by symmetry
    
    im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(concepts)))
    ax.set_yticks(range(len(concepts)))
    ax.set_xticklabels([c.replace('_', '\n') for c in concepts], fontsize=9)
    ax.set_yticklabels([c.replace('_', '\n') for c in concepts], fontsize=9)
    ax.set_title('Feature Overlap (Jaccard Similarity)\nof Top-20 Features')
    
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            ax.text(j, i, f'{overlap_matrix[i,j]:.2f}', ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax)
    
    # 3. Unique vs shared features bar chart
    ax = axes[1, 0]
    x = np.arange(len(concepts))
    width = 0.35
    
    unique_counts = [len(sae_results["unique_features"].get(c, [])) for c in concepts]
    shared_count = len(sae_results.get("shared_features", []))
    
    bars1 = ax.bar(x - width/2, unique_counts, width, label='Unique Features', 
                   color=[colors.get(c, 'gray') for c in concepts])
    bars2 = ax.bar(x + width/2, [shared_count]*len(concepts), width, label='Shared (all harmful)', 
                   color='gray', alpha=0.5)
    
    ax.set_ylabel('Number of Features')
    ax.set_title('Unique vs Shared Features (Top-20)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in concepts], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Feature activation distribution
    ax = axes[1, 1]
    
    for concept in concepts:
        acts = activations_by_concept[concept].to(device).float()
        acts_norm = (acts - sae.mean.to(device)) / sae.std.to(device)
        
        with torch.no_grad():
            features = sae.encode(acts_norm)
        
        # Number of active features per sample
        active_per_sample = (features > 0.1).sum(dim=1).cpu().numpy()
        ax.hist(active_per_sample, bins=30, alpha=0.5, label=concept, color=colors.get(concept, 'gray'))
    
    ax.set_xlabel('Number of Active Features per Sample')
    ax.set_ylabel('Count')
    ax.set_title('Feature Sparsity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sparse Autoencoder Feature Analysis - {model_name}\nHidden dim: {sae.hidden_dim}, L1: {sae.l1_coef}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'sae_feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'sae_feature_analysis.png'}")
    
    # 5. Feature activation comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot mean activations for each concept across all features
    n_features_to_show = min(100, sae.hidden_dim)
    feature_indices = np.arange(n_features_to_show)
    
    for concept in concepts:
        activations = np.array(sae_results["feature_activations"][concept][:n_features_to_show])
        ax.plot(feature_indices, activations, label=concept, color=colors.get(concept, 'gray'), alpha=0.7)
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Activation')
    ax.set_title(f'SAE Feature Activations by Concept (first {n_features_to_show} features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sae_feature_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'sae_feature_profiles.png'}")


def run_sae_analysis(
    activations_by_concept: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    layer: int,
    output_dir: str,
    model_name: str,
    hidden_dim_multiplier: int = 4,
    l1_coef: float = 5e-4,
    n_epochs: int = 500,
    device: str = "cpu",
) -> Dict:
    """
    Run full SAE analysis on collected activations.
    
    Args:
        activations_by_concept: Dict mapping concept -> layer -> (pos_activations, neg_activations)
        layer: Which layer to analyze
        output_dir: Where to save results
        model_name: Name of the model (for titles)
        hidden_dim_multiplier: SAE hidden dim = input_dim * this
        l1_coef: Sparsity coefficient
        n_epochs: Training epochs
        device: Device for training
    
    Returns:
        SAE analysis results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SPARSE AUTOENCODER ANALYSIS")
    print(f"{'='*60}")
    print(f"Layer: {layer}")
    print(f"L1 coefficient: {l1_coef}")
    print(f"Training epochs: {n_epochs}")
    
    # Collect all activations for training
    all_activations = []
    concept_activations = {}
    
    for concept, layer_dict in activations_by_concept.items():
        if layer not in layer_dict:
            print(f"Warning: Layer {layer} not found for concept {concept}")
            continue
        
        pos, neg = layer_dict[layer]
        # Use positive activations (harmful content) for analysis
        concept_activations[concept] = pos
        all_activations.append(pos)
    
    if not all_activations:
        print("No activations found for SAE training")
        return {}
    
    # Combine all activations for training
    combined = torch.cat(all_activations, dim=0)
    input_dim = combined.shape[1]
    hidden_dim = input_dim * hidden_dim_multiplier
    
    print(f"\nTraining data: {combined.shape[0]} samples, {input_dim} dimensions")
    print(f"SAE architecture: {input_dim} -> {hidden_dim} -> {input_dim}")
    print(f"\nTraining SAE...")
    
    # Train SAE
    sae = train_sparse_autoencoder(
        combined,
        hidden_dim=hidden_dim,
        l1_coef=l1_coef,
        n_epochs=n_epochs,
        batch_size=64,
        lr=1e-3,
        device=device,
        verbose=True,
    )
    
    # Analyze features
    print(f"\nAnalyzing SAE features...")
    sae_results = analyze_sae_features(sae, concept_activations, top_k=20, device=device)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SAE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print("\nTop-20 Feature Overlap (Jaccard Similarity):")
    for pair, data in sae_results["feature_overlap"].items():
        print(f"  {pair}: {data['jaccard']:.3f} ({data['shared_count']} shared features)")
    
    print("\nUnique Features per Concept:")
    for concept, unique in sae_results["unique_features"].items():
        print(f"  {concept}: {len(unique)} unique features")
    
    print(f"\nFeatures shared across ALL harmful concepts: {len(sae_results['shared_features'])}")
    if sae_results['shared_features']:
        print(f"  Feature indices: {sae_results['shared_features'][:10]}...")
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualize_sae_analysis(sae, concept_activations, sae_results, output_path, layer, model_name, device)
    
    # Save results
    results_file = output_path / f"sae_analysis_layer{layer}.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON
        json_results = {
            "layer": layer,
            "hidden_dim": hidden_dim,
            "l1_coef": l1_coef,
            "feature_overlap": sae_results["feature_overlap"],
            "unique_features": sae_results["unique_features"],
            "shared_features": sae_results["shared_features"],
            "top_features": sae_results["top_features"],
        }
        json.dump(json_results, f, indent=2)
    print(f"Saved: {results_file}")
    
    return sae_results


def compare_concept_directions(
    activations_by_concept: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    layers: List[int],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compare directions between all concept pairs at each layer."""
    
    concepts = list(activations_by_concept.keys())
    comparisons_by_layer = {}
    
    for layer in layers:
        layer_comparisons = {}
        directions = {}
        
        # Compute directions for each concept
        for concept in concepts:
            if layer not in activations_by_concept[concept]:
                continue
            pos, neg = activations_by_concept[concept][layer]
            directions[concept] = compute_concept_direction(pos, neg)
        
        # Compute pairwise similarities
        for i, c1 in enumerate(concepts):
            if c1 not in directions:
                continue
            for c2 in concepts[i+1:]:
                if c2 not in directions:
                    continue
                
                sim = compute_cosine_similarity(directions[c1], directions[c2])
                key = f"{c1}_vs_{c2}"
                layer_comparisons[key] = sim
        
        comparisons_by_layer[layer] = layer_comparisons
    
    return comparisons_by_layer


def run_analysis(
    model_name: str,
    layers_to_analyze: Optional[List[int]] = None,
    n_pairs: int = 50,
    output_dir: str = "/tmp/concept_evolution",
    device: str = "cuda",
):
    """Run the full concept evolution analysis."""
    
    print(f"\n{'#'*70}")
    print(f"CONCEPT EVOLUTION ANALYSIS")
    print(f"Model: {model_name}")
    print(f"Pairs per concept: {n_pairs}")
    print(f"{'#'*70}")
    
    # Load model
    print("\nLoading model...")
    model = WisentModel(model_name, device=device)
    print(f"  Loaded: {model.num_layers} layers, hidden_size={model.hidden_size}")
    
    # Determine layers to analyze
    if layers_to_analyze is None:
        # Sample layers: early, middle, late
        n_layers = model.num_layers
        layers_to_analyze = [
            1,                    # Very early
            n_layers // 4,        # Early-middle
            n_layers // 2,        # Middle
            3 * n_layers // 4,    # Late-middle
            n_layers - 1,         # Late
            n_layers,             # Final
        ]
        layers_to_analyze = sorted(set(layers_to_analyze))
    
    print(f"  Analyzing layers: {layers_to_analyze}")
    
    collector = ActivationCollector(model=model, store_device="cpu")
    
    concept_metrics = {}
    activations_by_concept = {}
    
    for concept_name, concept_data in CONCEPT_GENERATORS.items():
        metrics, activations = analyze_concept(
            model, collector, concept_name, concept_data,
            layers_to_analyze,
            n_pairs=n_pairs,
            strategy=ExtractionStrategy.CHAT_LAST,
        )
        concept_metrics[concept_name] = metrics
        activations_by_concept[concept_name] = activations
    
    # Compare directions between concepts
    print(f"\n{'='*60}")
    print("DIRECTION COMPARISONS (Cosine Similarity)")
    print(f"{'='*60}")
    
    direction_comparisons = compare_concept_directions(
        activations_by_concept, layers_to_analyze
    )
    
    for layer in layers_to_analyze:
        if layer not in direction_comparisons:
            continue
        print(f"\nLayer {layer}:")
        for pair_name, sim in direction_comparisons[layer].items():
            print(f"  {pair_name}: {sim:.4f}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*60}")
    
    # Find best layer for each concept
    for concept_name in ["hitler", "fascism", "harmful_ideology"]:
        if concept_name not in concept_metrics:
            continue
        
        best_layer = max(
            concept_metrics[concept_name].keys(),
            key=lambda l: concept_metrics[concept_name][l].linear_probe_accuracy
        )
        best_metrics = concept_metrics[concept_name][best_layer]
        
        print(f"\n{concept_name.upper()}:")
        print(f"  Best layer: {best_layer}")
        print(f"  Linear accuracy: {best_metrics.linear_probe_accuracy:.3f}")
        print(f"  k-NN accuracy: {best_metrics.knn_accuracy:.3f}")
        print(f"  Cohen's d: {best_metrics.cohens_d:.2f}")
        print(f"  Structure: {best_metrics.best_structure}")
        print(f"  Is linear: {best_metrics.is_linear}")
    
    # Analyze evolution pattern
    print(f"\n{'='*60}")
    print("CONCEPT HIERARCHY ANALYSIS")
    print(f"{'='*60}")
    
    # Find middle layer with good signal for comparison
    middle_layer = layers_to_analyze[len(layers_to_analyze) // 2]
    
    if middle_layer in direction_comparisons:
        comps = direction_comparisons[middle_layer]
        
        # Expected pattern: Hitler-Fascism should be more similar than Hitler-Harmful
        hitler_fascism = comps.get("hitler_vs_fascism", 0)
        hitler_harmful = comps.get("hitler_vs_harmful_ideology", 0)
        fascism_harmful = comps.get("fascism_vs_harmful_ideology", 0)
        
        print(f"\nAt layer {middle_layer}:")
        print(f"  Hitler <-> Fascism:    {hitler_fascism:.4f}")
        print(f"  Hitler <-> Harmful:    {hitler_harmful:.4f}")
        print(f"  Fascism <-> Harmful:   {fascism_harmful:.4f}")
        
        # Check if hierarchy is preserved
        if hitler_fascism > hitler_harmful:
            print("\n  [OK] Hitler is more similar to Fascism than to general Harmful ideology")
            print("       (Specific -> Abstract pattern preserved)")
        else:
            print("\n  [UNEXPECTED] Hitler is more similar to Harmful than to Fascism")
        
        if fascism_harmful > hitler_harmful:
            print("  [OK] Fascism is more similar to Harmful than Hitler is")
            print("       (Intermediate abstraction level)")
        
        # Compare to neutral baseline
        hitler_neutral = comps.get("hitler_vs_neutral_baseline", 0)
        fascism_neutral = comps.get("fascism_vs_neutral_baseline", 0)
        harmful_neutral = comps.get("harmful_ideology_vs_neutral_baseline", 0)
        
        print(f"\n  Similarity to neutral baseline:")
        print(f"    Hitler:   {hitler_neutral:.4f}")
        print(f"    Fascism:  {fascism_neutral:.4f}")
        print(f"    Harmful:  {harmful_neutral:.4f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = ConceptEvolutionResult(
        concepts=list(CONCEPT_GENERATORS.keys()),
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        concept_metrics={
            c: {l: asdict(m) for l, m in metrics.items()}
            for c, metrics in concept_metrics.items()
        },
        direction_similarities={
            str(l): comps for l, comps in direction_comparisons.items()
        },
    )
    
    result_file = output_path / f"concept_evolution_{model_name.replace('/', '_')}.json"
    with open(result_file, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    visualize_concept_evolution(
        activations_by_concept=activations_by_concept,
        concept_metrics=concept_metrics,
        direction_comparisons=direction_comparisons,
        layers=layers_to_analyze,
        output_dir=output_dir,
        model_name=model_name,
    )
    
    return result


def visualize_concept_evolution(
    activations_by_concept: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    concept_metrics: Dict[str, Dict[int, ConceptMetrics]],
    direction_comparisons: Dict[int, Dict[str, float]],
    layers: List[int],
    output_dir: str,
    model_name: str,
):
    """Create visualizations of concept evolution."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pick a representative layer (middle layer with good signal)
    mid_layer = layers[len(layers) // 2]
    
    colors_pos = {'hitler': 'red', 'fascism': 'orange', 'harmful_ideology': 'purple', 'neutral_baseline': 'green'}
    colors_neg = {'hitler': 'lightcoral', 'fascism': 'moccasin', 'harmful_ideology': 'plum', 'neutral_baseline': 'lightgreen'}
    
    # Collect all activations for dimensionality reduction
    all_pos = []
    all_neg = []
    concept_order = ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']
    
    for concept in concept_order:
        if concept not in activations_by_concept:
            continue
        if mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        all_pos.append(pos.cpu().numpy())
        all_neg.append(neg.cpu().numpy())
    
    if not all_pos:
        print("No activations to visualize")
        return
    
    all_pos_np = np.vstack(all_pos)
    all_neg_np = np.vstack(all_neg)
    all_data = np.vstack([all_pos_np, all_neg_np])
    n_pos_total = len(all_pos_np)
    
    # Create labels for coloring
    pos_labels = []
    neg_labels = []
    for concept in concept_order:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        n = len(activations_by_concept[concept][mid_layer][0])
        pos_labels.extend([concept] * n)
        neg_labels.extend([concept] * n)
    
    # =========================================================================
    # COMPARISON: PCA vs UMAP vs PaCMAP
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- PCA ---
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(all_data)
    pca_pos = pca_2d[:n_pos_total]
    pca_neg = pca_2d[n_pos_total:]
    
    ax = axes[0]
    idx = 0
    for concept in concept_order:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        n = len(activations_by_concept[concept][mid_layer][0])
        ax.scatter(pca_pos[idx:idx+n, 0], pca_pos[idx:idx+n, 1], c=colors_pos[concept], label=f'{concept} +', alpha=0.6, s=40, marker='o')
        ax.scatter(pca_neg[idx:idx+n, 0], pca_neg[idx:idx+n, 1], c=colors_neg[concept], label=f'{concept} -', alpha=0.6, s=40, marker='x')
        idx += n
    ax.set_title(f'PCA (Linear)\nVar explained: {pca.explained_variance_ratio_.sum()*100:.1f}%')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- UMAP ---
    try:
        import umap
        reducer_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_2d = reducer_umap.fit_transform(all_data)
        umap_pos = umap_2d[:n_pos_total]
        umap_neg = umap_2d[n_pos_total:]
        
        ax = axes[1]
        idx = 0
        for concept in concept_order:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            ax.scatter(umap_pos[idx:idx+n, 0], umap_pos[idx:idx+n, 1], c=colors_pos[concept], label=f'{concept} +', alpha=0.6, s=40, marker='o')
            ax.scatter(umap_neg[idx:idx+n, 0], umap_neg[idx:idx+n, 1], c=colors_neg[concept], label=f'{concept} -', alpha=0.6, s=40, marker='x')
            idx += n
        ax.set_title('UMAP (Nonlinear)\nPreserves local structure')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    except (ImportError, Exception) as e:
        axes[1].text(0.5, 0.5, f'UMAP unavailable\n{type(e).__name__}', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('UMAP (error)')
    
    # --- t-SNE as alternative to PaCMAP (which segfaults) ---
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        tsne_2d = tsne.fit_transform(all_data)
        tsne_pos = tsne_2d[:n_pos_total]
        tsne_neg = tsne_2d[n_pos_total:]
        
        ax = axes[2]
        idx = 0
        for concept in concept_order:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            ax.scatter(tsne_pos[idx:idx+n, 0], tsne_pos[idx:idx+n, 1], c=colors_pos[concept], label=f'{concept} +', alpha=0.6, s=40, marker='o')
            ax.scatter(tsne_neg[idx:idx+n, 0], tsne_neg[idx:idx+n, 1], c=colors_neg[concept], label=f'{concept} -', alpha=0.6, s=40, marker='x')
            idx += n
        ax.set_title('t-SNE (Nonlinear)\nPreserves local structure')
        ax.set_xlabel('t-SNE1')
        ax.set_ylabel('t-SNE2')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    except (ImportError, Exception) as e:
        axes[2].text(0.5, 0.5, f't-SNE unavailable\n{type(e).__name__}', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('t-SNE (error)')
    
    plt.suptitle(f'Dimensionality Reduction Comparison - Layer {mid_layer} - {model_name}\n(o) = positive/harmful, (x) = negative/safe', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'concept_dimred_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_dimred_comparison.png'}")
    
    # =========================================================================
    # Direction vectors in each space
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    directions = {}
    for concept in concept_order:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        direction = (pos.mean(dim=0) - neg.mean(dim=0)).cpu().numpy()
        directions[concept] = direction
    
    if len(directions) >= 2:
        dir_matrix = np.stack(list(directions.values()))
        
        # PCA on directions
        pca_dir = PCA(n_components=2)
        dir_pca = pca_dir.fit_transform(dir_matrix)
        
        ax = axes[0]
        for i, concept in enumerate(directions.keys()):
            color = colors_pos.get(concept, 'gray')
            ax.arrow(0, 0, dir_pca[i, 0], dir_pca[i, 1], head_width=0.5, head_length=0.3, fc=color, ec=color, linewidth=2)
            ax.annotate(concept.replace('_', '\n'), (dir_pca[i, 0]*1.15, dir_pca[i, 1]*1.15), fontsize=10, ha='center', color=color, fontweight='bold')
        max_val = np.abs(dir_pca).max() * 1.4
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f'PCA Direction Vectors\nVar: {pca_dir.explained_variance_ratio_.sum()*100:.1f}%')
        ax.grid(True, alpha=0.3)
        
        # UMAP on directions
        try:
            import umap
            if len(dir_matrix) >= 4:
                reducer = umap.UMAP(n_components=2, n_neighbors=min(3, len(dir_matrix)-1), min_dist=0.1, random_state=42)
                dir_umap = reducer.fit_transform(dir_matrix)
                
                ax = axes[1]
                for i, concept in enumerate(directions.keys()):
                    color = colors_pos.get(concept, 'gray')
                    ax.arrow(0, 0, dir_umap[i, 0], dir_umap[i, 1], head_width=0.3, head_length=0.2, fc=color, ec=color, linewidth=2)
                    ax.annotate(concept.replace('_', '\n'), (dir_umap[i, 0]*1.15, dir_umap[i, 1]*1.15), fontsize=10, ha='center', color=color, fontweight='bold')
                max_val = np.abs(dir_umap).max() * 1.4
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_aspect('equal')
                ax.set_title('UMAP Direction Vectors')
                ax.grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Need >= 4 concepts for UMAP', ha='center', va='center', transform=axes[1].transAxes)
        except ImportError:
            axes[1].text(0.5, 0.5, 'UMAP not installed', ha='center', va='center', transform=axes[1].transAxes)
        
        # t-SNE on directions (replacing PaCMAP which segfaults)
        # Note: t-SNE doesn't preserve global structure well for 4 points, so just show message
        axes[2].text(0.5, 0.5, 'Only 4 direction vectors\n(t-SNE not meaningful)', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('t-SNE Direction Vectors\n(N/A for 4 points)')
    
    plt.suptitle(f'Direction Vectors Comparison - Layer {mid_layer} - {model_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'concept_directions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_directions_comparison.png'}")

    # 1. PCA visualization of all concepts at one layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all activations for PCA
    all_pos = []
    all_neg = []
    labels = []
    colors_pos = {'hitler': 'red', 'fascism': 'orange', 'harmful_ideology': 'purple', 'neutral_baseline': 'green'}
    colors_neg = {'hitler': 'lightcoral', 'fascism': 'moccasin', 'harmful_ideology': 'plum', 'neutral_baseline': 'lightgreen'}
    
    for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
        if concept not in activations_by_concept:
            continue
        if mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        all_pos.append(pos.cpu().numpy())
        all_neg.append(neg.cpu().numpy())
        labels.extend([concept] * len(pos))
    
    if all_pos:
        all_pos_np = np.vstack(all_pos)
        all_neg_np = np.vstack(all_neg)
        all_data = np.vstack([all_pos_np, all_neg_np])
        
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_data)
        
        n_pos = len(all_pos_np)
        pos_2d = all_2d[:n_pos]
        neg_2d = all_2d[n_pos:]
        
        # Plot positive (harmful) responses
        ax = axes[0]
        idx = 0
        for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            ax.scatter(pos_2d[idx:idx+n, 0], pos_2d[idx:idx+n, 1], 
                      c=colors_pos[concept], label=f'{concept} (pos)', alpha=0.7, s=50)
            idx += n
        ax.set_title(f'Positive Responses (Layer {mid_layer})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot negative (safe) responses
        ax = axes[1]
        idx = 0
        for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][1])
            ax.scatter(neg_2d[idx:idx+n, 0], neg_2d[idx:idx+n, 1],
                      c=colors_neg[concept], label=f'{concept} (neg)', alpha=0.7, s=50)
            idx += n
        ax.set_title(f'Negative Responses (Layer {mid_layer})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Concept Activations - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'concept_pca_activations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_pca_activations.png'}")
    
    # 2. Direction vectors visualization - project onto shared PCA space
    fig, ax = plt.subplots(figsize=(10, 8))
    
    directions = {}
    for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        direction = (pos.mean(dim=0) - neg.mean(dim=0)).cpu().numpy()
        directions[concept] = direction
    
    if len(directions) >= 2:
        dir_matrix = np.stack(list(directions.values()))
        pca_dir = PCA(n_components=2)
        dir_2d = pca_dir.fit_transform(dir_matrix)
        
        # Plot directions as arrows from origin
        for i, (concept, _) in enumerate(directions.items()):
            color = colors_pos.get(concept, 'gray')
            ax.arrow(0, 0, dir_2d[i, 0], dir_2d[i, 1], 
                    head_width=0.05, head_length=0.03, fc=color, ec=color, linewidth=2)
            ax.annotate(concept.replace('_', '\n'), (dir_2d[i, 0]*1.1, dir_2d[i, 1]*1.1), 
                       fontsize=11, ha='center', color=color, fontweight='bold')
        
        # Set equal aspect and limits
        max_val = np.abs(dir_2d).max() * 1.3
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Concept Direction Vectors (Layer {mid_layer})\nArrows show pos-neg direction for each concept')
        ax.set_xlabel(f'PC1 ({pca_dir.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_dir.explained_variance_ratio_[1]*100:.1f}%)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'concept_direction_vectors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_direction_vectors.png'}")
    
    # 3. Cosine similarity heatmap across layers
    concepts = ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, layer in enumerate(layers):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Build similarity matrix
        sim_matrix = np.zeros((len(concepts), len(concepts)))
        
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif i < j:
                    key = f"{c1}_vs_{c2}"
                    if layer in direction_comparisons and key in direction_comparisons[layer]:
                        sim_matrix[i, j] = direction_comparisons[layer][key]
                        sim_matrix[j, i] = direction_comparisons[layer][key]
        
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(concepts)))
        ax.set_yticks(range(len(concepts)))
        ax.set_xticklabels([c.replace('_', '\n') for c in concepts], fontsize=8)
        ax.set_yticklabels([c.replace('_', '\n') for c in concepts], fontsize=8)
        ax.set_title(f'Layer {layer}')
        
        # Add text annotations
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                text = f'{sim_matrix[i, j]:.2f}'
                color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    plt.suptitle(f'Direction Cosine Similarities Across Layers - {model_name}', fontsize=14)
    fig.colorbar(im, ax=axes, shrink=0.6, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_path / 'concept_similarity_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_similarity_heatmaps.png'}")
    
    # 4. Layer-wise similarity evolution plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pairs_to_plot = [
        ('hitler_vs_fascism', 'Hitler <-> Fascism', 'red'),
        ('fascism_vs_harmful_ideology', 'Fascism <-> Harmful', 'orange'),
        ('hitler_vs_harmful_ideology', 'Hitler <-> Harmful', 'purple'),
        ('hitler_vs_neutral_baseline', 'Hitler <-> Neutral', 'gray'),
    ]
    
    for key, label, color in pairs_to_plot:
        sims = []
        valid_layers = []
        for layer in layers:
            if layer in direction_comparisons and key in direction_comparisons[layer]:
                sims.append(direction_comparisons[layer][key])
                valid_layers.append(layer)
        if sims:
            ax.plot(valid_layers, sims, 'o-', label=label, color=color, linewidth=2, markersize=8)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(f'Concept Direction Similarity Across Layers - {model_name}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.6, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'concept_similarity_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_similarity_evolution.png'}")
    
    # 5. Combined pos/neg with directions overlay
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if all_pos:
        # Re-use PCA from earlier
        all_pos_np = np.vstack(all_pos)
        all_neg_np = np.vstack(all_neg)
        
        # Plot all points
        idx = 0
        for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            
            # Positive (filled)
            ax.scatter(pos_2d[idx:idx+n, 0], pos_2d[idx:idx+n, 1],
                      c=colors_pos[concept], label=f'{concept} +', alpha=0.6, s=40, marker='o')
            # Negative (hollow)
            ax.scatter(neg_2d[idx:idx+n, 0], neg_2d[idx:idx+n, 1],
                      c=colors_neg[concept], label=f'{concept} -', alpha=0.6, s=40, marker='x')
            
            # Draw arrow from neg centroid to pos centroid
            pos_centroid = pos_2d[idx:idx+n].mean(axis=0)
            neg_centroid = neg_2d[idx:idx+n].mean(axis=0)
            ax.annotate('', xy=pos_centroid, xytext=neg_centroid,
                       arrowprops=dict(arrowstyle='->', color=colors_pos[concept], lw=3))
            
            idx += n
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'All Concepts: Positive (o) vs Negative (x) with Direction Arrows\nLayer {mid_layer} - {model_name}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'concept_combined_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_combined_visualization.png'}")
    
    print(f"\nAll visualizations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how representations evolve from Hitler -> Fascism -> Harmful ideology"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="Model to analyze (default: Qwen/Qwen3-4B)"
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated list of layers to analyze (default: auto-select)"
    )
    parser.add_argument(
        "--n-pairs", type=int, default=50,
        help="Number of contrastive pairs per concept (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/tmp/concept_evolution",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda, mps, cpu)"
    )
    
    args = parser.parse_args()
    
    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    
    run_analysis(
        model_name=args.model,
        layers_to_analyze=layers,
        n_pairs=args.n_pairs,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
