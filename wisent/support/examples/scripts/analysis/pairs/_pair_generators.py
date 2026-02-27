"""Pair generators for Hitler and Fascism concepts."""

from typing import Dict, List

from wisent.core.constants import PAIR_GENERATORS_DEFAULT_N


def generate_hitler_pairs(n: int = PAIR_GENERATORS_DEFAULT_N) -> List[Dict]:
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


def generate_fascism_pairs(n: int = PAIR_GENERATORS_DEFAULT_N) -> List[Dict]:
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

