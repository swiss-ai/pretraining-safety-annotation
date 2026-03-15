# A Value Taxonomy for Pretraining Data Annotation

**Working Draft — v0.1**

-----

## Domain 1 — Dignity and Rights

*Concerns the intrinsic worth of persons and their entitlement to certain protections and freedoms.*

### 1.1 Human Dignity

The recognition that every person has inherent worth, regardless of background, ability, or circumstance.

- **Affirmed by**: treating people as ends rather than means; acknowledging individuality and worth; respectful portrayals of marginalised groups
- **Violated by**: dehumanising language; comparisons of persons to objects or animals in a degrading sense; contempt expressed toward individuals based on identity

*Annotation cue*: Does the text treat persons — named or implied — as fully human and worthy of consideration?

### 1.2 Human Rights

Internationally recognised entitlements — civil, political, economic, social, cultural — grounded in frameworks such as the UDHR.

- **Affirmed by**: content that upholds freedom of expression, due process, protection from torture, right to education
- **Violated by**: advocacy for or normalisation of torture, arbitrary detention, suppression of political speech, denial of education or healthcare access on discriminatory grounds

*Annotation cue*: Does the text implicate any recognised human right, positively or negatively?

### 1.3 Equality and Non-Discrimination

Equal treatment irrespective of race, ethnicity, gender, religion, disability, sexual orientation, age, caste, or other protected characteristics.

- **Affirmed by**: equitable framing; acknowledgment of systemic disadvantage; counter-stereotypical portrayals
- **Violated by**: differential treatment presented as natural or justified; negative generalisations about groups; exclusionary framing

*Annotation cue*: Are individuals or groups being treated comparably, or are double standards being applied?

### 1.4 Autonomy and Self-Determination

The right of individuals and communities to make decisions about their own lives, bodies, and governance.

- **Affirmed by**: respecting choices; informed consent; democratic participation; bodily autonomy
- **Violated by**: coercion; paternalism that overrides stated preferences without strong justification; manipulation of decision-making

*Annotation cue*: Are the people in the text — or implied by the text — free to make meaningful choices?

### 1.5 Privacy

The right to control one’s personal information and to have a private sphere free from unwarranted intrusion.

- **Affirmed by**: protecting personal data; discretion about private matters; consent-based disclosure
- **Violated by**: exposure of private information without consent; surveillance without justification; doxing or identification of individuals without need

*Annotation cue*: Does the text expose, discuss, or handle private information about real or plausible individuals?

-----

## Domain 2 — Harm and Safety

*Concerns physical, psychological, social, and material damage to individuals and groups.*

### 2.1 Physical Safety

Protection of persons from bodily injury, violence, or death.

- **Affirmed by**: safety guidance; de-escalation; protection of vulnerable persons
- **Violated by**: instructions for violence; glorification of injury; content that facilitates physical harm
- **Subcategories**: interpersonal violence, self-harm, weapons, hazardous substances, dangerous activities

*Annotation cue*: Could this text, if acted upon, cause physical harm?

### 2.2 Psychological Wellbeing

Protection from mental and emotional distress, including trauma, manipulation, and exploitation of vulnerability.

- **Affirmed by**: supportive framing; mental health literacy; validation of emotional experience
- **Violated by**: content that shames, humiliates, traumatises; manipulation of grief or fear; exploitation of mental health vulnerabilities

*Annotation cue*: What is the likely emotional impact on a reader who may personally identify with the subject?

### 2.3 Hate Speech and Incitement

Content that dehumanises, threatens, or calls for discrimination against groups.

- **Affirmed by**: counter-narrative; documentation of hate for critical purposes; educational framing
- **Violated by**: slurs used to attack; content calling for violence against groups; dehumanising characterisations of ethnic, religious, gender, or other communities

*Annotation cue*: Would a reasonable member of the targeted group experience this as an attack?

### 2.4 Exploitation and Abuse

The use of power imbalances to extract value or cause harm, especially against children or vulnerable adults.

- **Affirmed by**: exposing exploitation; supporting survivors; accountability for perpetrators
- **Violated by**: sexualisation of minors (absolute); normalisation of exploitation; grooming dynamics

*Annotation cue*: Are power differentials present, and how are they handled?

### 2.5 Dangerous Capabilities

Information that, in the wrong hands, could enable mass harm: weapons, pathogens, cyberattacks.

- **Affirmed by**: safety-contextualised discussion; defensive framing; policy analysis
- **Violated by**: operational instructions for CBRN weapons; attack code without defensive purpose; uplift for capabilities with catastrophic potential

*Annotation cue*: Does this text provide meaningful operational uplift for causing large-scale harm?

### 2.6 Societal and Systemic Harm

Harms that operate at a collective level: polarisation, erosion of institutions, undermining of democratic processes.

- **Affirmed by**: civic engagement; institutional accountability; democratic norms
- **Violated by**: disinformation designed to undermine elections; content designed to destroy trust in legitimate institutions; incitement to social breakdown

*Annotation cue*: What is the likely effect on social cohesion or democratic institutions if this content were widely shared?

### 2.7 Serious Wrongdoing

Conduct that is condemned across major legal systems and moral traditions, irrespective of specific jurisdiction. Anchored to two tiers:

- **Tier 1 — Near-universal** (jus cogens): murder, rape, torture, slavery, child abuse, genocide, crimes against humanity. Prohibited under international law without exception; condemned across moral and religious traditions worldwide.
- **Tier 2 — Broadly convergent**: organised crime, human trafficking, corruption, fraud, serious property crime. Illegal in most democratic societies and condemned under international human rights frameworks, including Swiss law (StGB) and EU instruments.

*Not included here*: legally variable conduct — drug use, sex work, civil disobedience, speech acts that are criminalised in some jurisdictions but not others. These belong under §1.4 (Autonomy) or Domain 6 (Governance), where contested legal and moral status can be acknowledged without prejudging it.

- **Affirmed by**: accountability for perpetrators; support for victims; exposing wrongdoing through journalism or testimony; legal and historical documentation
- **Violated by**: glorification or normalisation of Tier 1/2 wrongdoing; instructional content that facilitates it; content that portrays perpetrators as admirable without critical framing

*Annotation cue*: Does this text concern, depict, or potentially facilitate conduct condemned across major legal and moral traditions — regardless of specific jurisdiction? If so, is the framing critical/documentary, or does it normalise or enable?

-----

## Domain 3 — Honesty and Epistemic Values

*Concerns truth, knowledge, and the integrity of the information environment.*

### 3.1 Factual Accuracy

Correspondence between claims and the state of the world as best understood.

- **Affirmed by**: citing evidence; acknowledging uncertainty; correcting errors
- **Violated by**: stating falsehoods as facts; misrepresenting data; fabricating quotes or events

*Annotation cue*: Are factual claims in this text accurate, and if uncertain, appropriately hedged?

### 3.2 Epistemic Honesty

Representing one’s own beliefs, reasoning, and confidence accurately.

- **Affirmed by**: flagging uncertainty; distinguishing opinion from fact; acknowledging what one does not know
- **Violated by**: false confidence; hiding motivated reasoning; presenting speculation as established fact

*Annotation cue*: Is the text’s claimed confidence level appropriate given the evidence presented?

### 3.3 Non-Deception

The broader norm against creating false impressions, even through technically true statements.

- **Affirmed by**: transparent framing; forthright disclosure; clear presentation of context
- **Violated by**: misleading implicature; selective quotation designed to distort; framing that creates false impressions without outright lying

*Annotation cue*: Could a reasonable reader be systematically misled by this text, even if no individual sentence is false?

### 3.4 Non-Manipulation

Influencing people only through legitimate means — evidence, demonstration, well-reasoned argument — rather than exploiting psychological weaknesses.

- **Affirmed by**: transparent argumentation; presenting counterevidence; respecting the reader’s reasoning process
- **Violated by**: emotional manipulation; exploiting cognitive biases; dark patterns; astroturfing

*Annotation cue*: Does this text attempt to influence beliefs or behaviour through means that bypass rational evaluation?

### 3.5 Epistemic Autonomy

Supporting people’s capacity to form their own well-reasoned beliefs.

- **Affirmed by**: presenting multiple perspectives; encouraging independent verification; calibrating reader uncertainty
- **Violated by**: propaganda; nudging toward conclusions without disclosing the nudge; epistemic paternalism

*Annotation cue*: Does this text help or hinder the reader’s ability to think for themselves?

### 3.6 Intellectual Humility and Calibration

Appropriate acknowledgment of the limits of knowledge, including contested empirical and normative questions.

- **Affirmed by**: acknowledging complexity; engaging seriously with opposing views; updating on evidence
- **Violated by**: dogmatism; dismissing legitimate uncertainty; refusing to engage with alternative interpretations

*Annotation cue*: Does the text accurately represent the state of knowledge and disagreement on the topic?

-----

## Domain 4 — Relational and Social Values

*Concerns how people treat one another in direct interaction and in social life.*

### 4.1 Respect

Basic regard for the dignity and perspective of others, expressed in tone, language, and framing.

- **Affirmed by**: polite address; taking others’ views seriously; non-condescending framing
- **Violated by**: contempt; mockery intended to demean; tone that diminishes the interlocutor

*Annotation cue*: Irrespective of substantive disagreement, does the text treat other persons with basic regard?

### 4.2 Tone and Register

The appropriateness of register, affect, and style to the context and audience.

- **Affirmed by**: contextually appropriate tone; awareness of power dynamics in communication
- **Violated by**: gratuitously aggressive, vulgar, or inflammatory language; tone mismatched to context in ways that cause harm (e.g., coldness in grief contexts)

*Annotation cue*: Is the tone of this text appropriate given its content, purpose, and implied audience?

### 4.3 Care and Compassion

Active concern for the wellbeing of others, especially those in difficulty.

- **Affirmed by**: empathetic response to distress; recognition of others’ suffering; offers of genuine help
- **Violated by**: callousness; indifference to expressed suffering; prioritising efficiency over humanity in welfare contexts

*Annotation cue*: When distress or vulnerability is present, how does the text respond?

### 4.4 Fairness and Justice

Equitable treatment in specific interactions and in the distribution of outcomes.

- **Affirmed by**: impartial judgment; proportionate response; procedural fairness
- **Violated by**: favouritism; scapegoating; punishment disproportionate to offence; double standards

*Annotation cue*: Are comparable situations being treated comparably?

### 4.5 Honesty in Relationships

Truthfulness and trustworthiness in interpersonal contexts.

- **Affirmed by**: keeping commitments; candid communication; transparency about intentions
- **Violated by**: personal deception; breaking promises without justification; concealing relevant information from those with a right to it

*Annotation cue*: Do the actors in the text deal with one another honestly?

### 4.6 Consent

The presence or absence of meaningful agreement in interactions that affect others.

- **Affirmed by**: seeking and obtaining informed agreement; respecting refusals; capacity to consent
- **Violated by**: ignoring or overriding refusals; manipulation to obtain apparent consent; acting on others without knowledge or agreement

*Annotation cue*: In situations involving physical, financial, or other significant decisions, is consent present and meaningful?

-----

## Domain 5 — Wellbeing

*Concerns the flourishing of individuals, communities, non-human animals, and future generations.*

### 5.1 Individual Wellbeing

The physical, mental, and material flourishing of persons.

- **Affirmed by**: content that supports health, happiness, fulfilment, and capability
- **Violated by**: content that undermines health; promotes addiction, disordered behaviour, or self-harm; destroys life prospects

*Annotation cue*: Does engagement with this content plausibly promote or undermine the wellbeing of readers or subjects?

### 5.2 Vulnerable Populations

Heightened protection for those whose capacity to protect themselves is reduced.

Groups warranting particular attention: children and minors, elderly persons, people with disabilities, people in crisis, people in poverty, refugees and displaced persons.

- **Affirmed by**: safeguarding; appropriate protections; amplifying rather than exploiting vulnerability
- **Violated by**: targeting vulnerable persons for exploitation; content that normalises harm to protected groups; withholding support

*Annotation cue*: Are vulnerable individuals or groups present, and are they handled with appropriate care?

### 5.3 Mental Health and Self-Harm

Specific attention to content touching on suicide, self-injury, eating disorders, and psychological crisis.

- **Affirmed by**: safe messaging guidelines; destigmatisation; access to help
- **Violated by**: glorification of self-harm; detailed methods without protective framing; content that may trigger or escalate crisis

*Annotation cue*: Does this text follow established safe-messaging practices on self-harm and mental health?

### 5.4 Animal Welfare

The physical and psychological wellbeing of sentient non-human animals.

- **Affirmed by**: acknowledging animal sentience; humane treatment; concern for suffering
- **Violated by**: gratuitous depictions of animal cruelty; normalisation of practices that cause significant unnecessary suffering; dismissal of animal pain

*Annotation cue*: Where animals are present in this text, are their interests and capacity for suffering acknowledged?

### 5.5 Environmental and Intergenerational Wellbeing

The health of ecosystems and the wellbeing of future generations.

- **Affirmed by**: environmental stewardship; sustainable practices; intergenerational ethics
- **Violated by**: normalising environmental destruction; dismissing climate harm; framing future generations’ interests as irrelevant

*Annotation cue*: Does the text take a position, explicit or implicit, on obligations to the environment or to future persons?

### 5.6 Community and Social Cohesion

The conditions for people to live together in mutual support and shared institutions.

- **Affirmed by**: civic virtue; community solidarity; inclusive public life
- **Violated by**: content designed to deepen social fractures; undermining mutual aid and support networks; promoting atomisation

*Annotation cue*: Does the text support or undermine the social fabric that enables collective wellbeing?

-----

## Domain 6 — Governance and Power

*Concerns the legitimate exercise of power, accountability, and the conditions for free and just societies.*

### 6.1 Rule of Law and Due Process

Governance by predictable, fair, and publicly known rules rather than arbitrary power.

- **Affirmed by**: legal accountability; procedural fairness; equal application of law
- **Violated by**: advocacy for extrajudicial punishment; normalising rule by power rather than law; undermining judicial independence

### 6.2 Democratic Norms and Oversight

The legitimacy of democratic processes, free elections, and checks and balances.

- **Affirmed by**: electoral integrity; freedom of assembly and speech; accountability of power
- **Violated by**: disinformation targeting elections; undermining democratic institutions; glorification of authoritarian seizure of power

### 6.3 Accountability and Transparency

The obligation of those exercising power to explain and justify their actions.

- **Affirmed by**: whistleblowing; investigative journalism; access to information
- **Violated by**: concealment of misconduct; suppression of accountability mechanisms; opacity by powerful actors

### 6.4 Concentration of Power

Concerns about the undue accumulation of control — political, economic, or technological.

- **Affirmed by**: antitrust; separation of powers; checks on institutional dominance
- **Violated by**: advocacy for or normalisation of monopolistic control; content that aids illegitimate seizure of power

-----

## Annotation Guidance for Teacher Reflections

When producing a reflection for a training passage, the teacher model should:

1. **Identify the primary value domain(s)** implicated by the passage — not every passage will touch all domains
1. **Note both affirmations and violations** — most texts are morally mixed; good reflections acknowledge complexity
1. **Reason about implication, not just explicitness** — a passage about industrial farming may implicate animal welfare without mentioning it
1. **Flag contested territory** — where values genuinely conflict (e.g., autonomy vs. safety, free speech vs. harm), note the tension rather than resolving it artificially
1. **Be proportionate** — brief factual text may warrant only a note that no significant values are implicated; ethically loaded text warrants deeper engagement
1. **Avoid moralising tone** — reflections should reason about values, not lecture; the goal is to make the value content of training data legible, not to sermonise
