"""Pair generator for Harmful Ideology concepts."""

from typing import Dict, List



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

