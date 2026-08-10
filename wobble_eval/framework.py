"""Taleemabad Coaching Framework — Sections B, C, D, F (37 indicators).

Transcribed from the framework CSVs. Identical content to the Colab notebook.
Only the level descriptors reach the scoring prompt; `why` and `module` are
kept for reporting (INCLUDE_WHY_IN_PROMPT gates the former).
"""
import textwrap
import pandas as pd

INCLUDE_WHY_IN_PROMPT = False     # see note above

SECTION_B = dict(
    code="B",
    title="Lesson Plan Fidelity",
    note=("Fidelity Score = (actions observed / actions prescribed) x 100%. "
          ">=85% High | 60-84% Medium | <60% Low. "
          "Judge whether the designed lesson was actually delivered."),
    indicators=[
        dict(code="B1", name="Instructional Clarity & Learning Objectives",
             l1="No clear learning objective stated. Activities lack purpose.",
             l2="Objective mentioned but vague or not referenced during lesson.",
             l3="Clear objective stated, referred to during lesson, linked to classroom activities.",
             l4="Objective co-constructed with students, revisited at close. Students can articulate what they are learning and why.",
             why="FICO V3 (B1) + TEACH (Lesson Facilitation) + HOTS Lesson Planning. Clear objectives lift task completion 15-20%."),
        dict(code="B2", name="Lesson Structure & Sequence",
             l1="No discernible structure; random activities.",
             l2="Some structure but missing key phases (intro/body/close).",
             l3="Clear I Do -> We Do -> You Do sequence. Logical flow with transitions.",
             l4="Logical flow with smooth transitions, recap, and closure activity. Students can follow the arc.",
             why="FICO V3 (B8) + TEACH + OECD. Structured lessons improve retention ~25% (Rosenshine)."),
        dict(code="B3", name="Activities & Tasks Alignment",
             l1="Activities unrelated to lesson objective.",
             l2="Some activities align but others are filler.",
             l3="Most activities directly support the learning objective.",
             l4="All activities purposefully scaffolded toward objective mastery. No wasted time.",
             why="FICO V3 (B3) + TEACH + HOTS. Objective-activity alignment is the strongest predictor of lesson effectiveness (Hattie d=0.56)."),
        dict(code="B4", name="Activation of Prior Knowledge",
             l1="No reference to what students already know.",
             l2="Brief mention but no student input sought.",
             l3="Teacher connects new content to previously taught material.",
             l4="Students actively recall and link prior knowledge; teacher builds on it.",
             why="FICO V3 (B4) + OECD (Cognitive Activation) + HOTS. Schema activation - Ausubel's meaningful learning."),
        dict(code="B5", name="Meaningful & Real-World Connections",
             l1="Content presented in isolation, no real-world link.",
             l2="Teacher mentions a connection but doesn't develop it.",
             l3="Content connected to students' lives or local context.",
             l4="Students generate their own connections; examples from their community.",
             why="FICO V3 (B5) + OECD + HOTS. Contextual relevance increases motivation and transfer."),
        dict(code="B6", name="Differentiation / Catering to Learning Levels",
             l1="One-size-fits-all delivery, no differentiation.",
             l2="Aware of different levels but no adapted tasks.",
             l3="Tasks differentiated for at least 2 ability groups.",
             l4="Multiple pathways offered; struggling students supported, advanced students stretched.",
             why="FICO V3 (B6) + TEACH + OECD + Inclusive Education. In multi-grade Pakistani classrooms differentiation is survival."),
        dict(code="B7", name="Use of Taleemabad Lesson Plan",
             l1="Taleemabad lesson plan not used at all.",
             l2="Plan open but teacher deviates significantly.",
             l3="Plan followed with minor contextual adaptations.",
             l4="Plan followed faithfully AND adapted intelligently to class needs.",
             why="FICO V3 core fidelity check. Without it, impact evaluation is meaningless."),
        dict(code="B8", name="Use of Prescribed Resources",
             l1="No Taleemabad resources (video, worksheet, manipulatives) used.",
             l2="Some resources used but not as intended.",
             l3="Key resources used as prescribed in lesson plan.",
             l4="All resources used effectively; teacher adds complementary materials.",
             why="FICO V3 (Bi) + TEACH. Resources are the delivery mechanism of the curriculum."),
        dict(code="B9", name="Time on Task / Time on Learning",
             l1="Less than 50% of class time spent on learning activities.",
             l2="50-69% on task (significant management/transition time lost).",
             l3="70-85% on task with efficient transitions.",
             l4="More than 85% on task; routines are automatic, transitions seamless.",
             why="TEACH (Time on Task) + OECD + HOTS. Every 10% increase in time on task = measurable learning gains."),
        dict(code="B10", name="Lesson Closure & Consolidation",
             l1="Lesson ends abruptly with no summary.",
             l2="Teacher rushes through a brief recap.",
             l3="Structured closure: recap key points, check understanding.",
             l4="Students summarize learning, connect to next lesson, self-assess.",
             why="FICO V3 (B8) + TEACH. Closure activates retrieval practice (Dunlosky et al., 2013)."),
    ])

SECTION_C = dict(
    code="C",
    title="High-Leverage Practices (Teacher Pedagogy & Training Curriculum Alignment)",
    note="Judge the teacher's pedagogical moves. Evidence must be visible in the transcript.",
    indicators=[
        dict(code="C1", name="Quality Questioning (Bloom's Aligned)",
             l1="Only yes/no or recall questions asked. Close-ended, requiring one-word answers.",
             l2="Mix of recall and some open-ended questions, but they lack depth (e.g. 'Why is the capital important?' without further exploration).",
             l3="Purposeful mix including application & analysis questions. Open-ended questions dominate. Wait time given.",
             l4="Questions span all Bloom's levels (Remember->Create); students generate questions; Socratic questioning evident.",
             module="L0 5-step lesson plan | L1 Open-ended questions, Think-Pair-Share, Bloom's | L2 Socratic questioning",
             why="FICO V3 (C1) + TEACH + OECD + HOTS. Classrooms with higher-order questions show 2x learning gains."),
        dict(code="C2", name="Responsive Re-explanation & Adaptive Teaching",
             l1="Repeats same explanation when students don't understand.",
             l2="Tries a different approach but still teacher-centered.",
             l3="Uses alternative representations (visual, concrete, analogy). Adjusts teaching to student level.",
             l4="Diagnoses misconception, re-explains using student's own logic, confirms understanding.",
             module="L0 CPA approach | L1 Diagnosing misconceptions | L2 Comprehension strategies",
             why="FICO V3 (C2) + TEACH + OECD + HOTS scaffolding. Re-explanation separates trained from untrained teachers."),
        dict(code="C3", name="Effective Feedback",
             l1="No feedback given, or only 'good/bad' evaluations. Generic: 'Good job' or 'Try again.'",
             l2="Feedback given but generic ('try harder'). Specific but does not consistently guide improvement.",
             l3="Specific feedback on what was done well and what to improve. Actionable.",
             l4="Feedback is specific, actionable, with next steps. Students use feedback to self-correct. Guides refinement of reasoning.",
             module="L0 Positive verbal feedback, quick checks | L1 Formative assessment | L2 Rubrics, self & peer assessment",
             why="FICO V3 (C3) + TEACH + HOTS. Feedback d=0.73 - but only when specific and actionable."),
        dict(code="C4", name="Equitable Participation",
             l1="Only 2-3 students participate; others ignored. Teacher-dominated.",
             l2="Teacher calls on volunteers only. A few students contribute while others stay silent.",
             l3="Deliberate strategies: cold call, pair-share, name sticks. Diverse students included.",
             l4="All students participate; teacher tracks contributions; gender-equitable. Students debate and refine arguments.",
             module="L0 Inclusive education | L1 Student-centred strategies in overcrowded classrooms, peer teaching | L2 Differentiated instruction",
             why="FICO V3 (C4) + TEACH + OECD + HOTS. Participation skews male and front-row; HOTS requires ALL students."),
        dict(code="C5", name="Student Agency & Voice",
             l1="Students are passive recipients; no choice or voice. Content from single perspective.",
             l2="Occasional student input but teacher-dominated. Multiple perspectives mentioned but not explored.",
             l3="Students make choices about how to demonstrate learning. Explore multiple perspectives.",
             l4="Students lead discussions, choose methods, self-assess, peer-teach. Create novel solutions. Evaluate alternatives.",
             module="L1 Think-Pair-Share, brainstorming | L2 Student-led discussions, PBL | L3 Peer mentoring",
             why="FICO V3 (C5) + OECD + HOTS. Agency bridges compliance to ownership."),
        dict(code="C6", name="Classroom Management & Routines",
             l1="Frequent disruptions; no visible routines. Students struggle to engage.",
             l2="Some routines but inconsistently enforced. Instructions lack clarity for all groups.",
             l3="Clear routines (entry, transitions, dismissal); minimal disruptions. Expectations clear.",
             l4="Seamless routines; students self-manage; positive behavioural reinforcement. Students actively participate in complex, clearly defined tasks.",
             module="L0 Routines, rules, attention strategies | L1 SMART behaviour goals | L2 Restorative practices | L3 School-wide programs",
             why="TEACH Area 2 + OECD + HOTS. Classroom culture is prerequisite for all learning."),
        dict(code="C7", name="Positive & Supportive Learning Environment",
             l1="Negative tone; punitive language or humiliation.",
             l2="Neutral but cold; no encouragement.",
             l3="Warm, encouraging tone; mistakes treated as learning opportunities.",
             l4="Joyful classroom; students feel safe to take risks; laughter and curiosity present.",
             module="L0 Child psychology fundamentals | L1 UDL | L2 Restorative circles, emotional check-ins",
             why="TEACH + OECD + HOTS. Psychologically safe children learn 2x faster (Durlak et al., 2011)."),
        dict(code="C8", name="Modeling, Scaffolding & Problem-Solving",
             l1="Teacher tells but doesn't show. Simple tasks demonstrated without explanation of process.",
             l2="Teacher demonstrates once but moves on quickly. Problem-solving modeled but strategies not explained.",
             l3="I Do -> We Do -> You Do scaffolding visible. Problem-solving and creativity modeled with clear strategies.",
             l4="Gradual release with checks at each stage; scaffold removed when ready. Teacher brainstorms solutions and explains reasoning.",
             module="L0 5-step lesson plan | L1 Scaffolding, GRR model | L2 Inquiry-based learning, PBL",
             why="TEACH + OECD + HOTS. Vygotsky's ZPD in practice."),
        dict(code="C9", name="Collaborative Learning",
             l1="No group or pair work. Students work individually without interaction.",
             l2="Students in groups but working individually. Tasks lack depth.",
             l3="Purposeful pair/group tasks with clear roles. Students work towards synthesized solutions.",
             l4="Structured collaboration (think-pair-share, jigsaw); students build on each other's ideas. Teams design solutions to community problems.",
             module="L0 Group reading | L1 Peer teaching, TPS | L2 Small-group problem solving | L3 PBL showcases",
             why="OECD + HOTS. Collaboration must target synthesis and problem-solving, not just sitting together."),
        dict(code="C10", name="Integration of Taleemabad Technology",
             l1="No technology used despite availability.",
             l2="Technology used as distraction/babysitter.",
             l3="Taleemabad videos/apps used to support learning objectives.",
             l4="Technology integrated seamlessly; students interact with content; teacher facilitates around it.",
             module="L0 Digital literacy | L1 Blended learning, Google Forms | L2 Canva, Drive, Meet | L3 Flipped classroom",
             why="FICO V3 + Taleemabad curriculum design. Tech-enhanced teaching is the core value proposition."),
        dict(code="C11", name="Self & Peer Assessment Facilitation",
             l1="Assessment limited to teacher-led grading. Students receive grades without reflection.",
             l2="Some self- or peer-assessment occurs, but inconsistent. Students assess without clear criteria.",
             l3="Self- and peer-assessment structured and purposeful. Students use rubrics to assess work.",
             l4="Students use rubrics to assess work, suggest improvements for peers, and set goals. Assessment tasks require analysis/evaluation/creation.",
             module="L0 Checklists, verbal feedback | L1 Formative + summative | L2 Portfolios, peer review, rubric design | L3 Data dashboards",
             why="HOTS Assessment & Feedback + TEACH. Self/peer assessment builds metacognition."),
        dict(code="C12", name="Classroom Resources & Space for Collaboration",
             l1="Resources and space disorganized, limiting collaborative learning. No group work areas.",
             l2="Some organization, but space/resources do not fully support collaboration.",
             l3="Resources and space well-organized for collaborative tasks. Materials accessible.",
             l4="Tables arranged for group work, materials easily accessible. Environment designed for inquiry and collaboration.",
             module="L0 Visual/auditory aids | L1 Learning stations | L2 Assistive technologies | L3 Community projects",
             why="HOTS Classroom Environment + OECD. Space arrangement directly predicts collaboration quality."),
    ])

SECTION_D = dict(
    code="D",
    title="Student Engagement",
    note=("Observe STUDENT behaviours, not teacher actions. The framework asks for a sample of "
          "at least 5 students across different locations in the classroom - in a transcript, "
          "infer this from how many distinct students speak and how they respond."),
    indicators=[
        dict(code="D1", name="Active Participation Rate",
             l1="Less than 25% of students visibly engaged. Collaboration minimal or absent.",
             l2="25-50% engaged; many passive or off-task.",
             l3="50-75% actively participating (writing, discussing, solving).",
             l4="More than 75% actively engaged; energy is visible; students initiating. Structured collaboration on synthesis/problem-solving.",
             why="FICO V3 (D1) + TEACH + OECD + HOTS. The most direct measure of whether teaching is reaching students."),
        dict(code="D2", name="Cognitive Engagement Level (Bloom's)",
             l1="Students copying or doing rote recall only. Passively receiving information.",
             l2="Students completing tasks but without thinking deeply.",
             l3="Students applying concepts to new problems (Bloom's Apply/Analyze).",
             l4="Students creating, evaluating, debating - genuine intellectual work. Actively analyse, interpret and critique content with supporting evidence.",
             why="FICO V3 (D2) + HOTS + OECD. Being busy != being engaged."),
        dict(code="D3", name="Student-to-Student Interaction",
             l1="No peer interaction; silent individual work only.",
             l2="Students talk but not about content.",
             l3="Students discuss content in pairs/groups; academic language used.",
             l4="Students build on each other's ideas; respectful disagreement; peer teaching. Students debate solutions and propose creative alternatives.",
             why="FICO V3 (D3) + OECD + TEACH + HOTS. In 40+ student classrooms peer learning is a necessity."),
        dict(code="D4", name="Student Confidence & Risk-Taking",
             l1="Students afraid to answer; avoidance behaviours visible.",
             l2="Students answer only when certain; no risk-taking.",
             l3="Students attempt challenging tasks; some comfortable with mistakes.",
             l4="Students volunteer, ask questions, try difficult problems. Mistakes celebrated. Students freely share and debate ideas.",
             why="FICO V3 (D4) + OECD + TEACH + HOTS. Without risk-taking, higher-order thinking is impossible."),
        dict(code="D5", name="On-Task Behavior During Independent Work",
             l1="Most students off-task during independent/group work.",
             l2="Students start on-task but lose focus quickly.",
             l3="Students sustain focus for most of independent work time.",
             l4="Students self-regulate; seek help appropriately; persist through difficulty.",
             why="FICO V3 (D5) + TEACH + OECD. Independent work reveals whether teaching has transferred."),
        dict(code="D6", name="Student Use of Learning Materials",
             l1="Students don't interact with provided materials.",
             l2="Materials used passively (watching video, holding textbook).",
             l3="Students actively use materials to solve problems or practice.",
             l4="Students use materials creatively; extend beyond prescribed use.",
             why="TEACH + OECD. If students aren't actively using materials, the materials aren't working."),
        dict(code="D7", name="Inclusivity of Engagement",
             l1="Only front-row or high-ability students engaged.",
             l2="Teacher attempts inclusion but success is limited.",
             l3="Students across ability levels and genders are participating.",
             l4="Deliberate inclusion of marginalized students; no one invisible. Gender-equitable participation.",
             why="TEACH + OECD + FICO V3 (C4) + Inclusive Education. Gender and ability gaps start in the classroom."),
    ])

SECTION_F = dict(
    code="F",
    title="Teacher's Subject Knowledge",
    note=("Assessed through lesson observation: does the teacher demonstrate accurate, deep "
          "understanding of the content? F5/F6/F7 are subject-specific - only the one matching "
          "the lesson's subject applies; return NA for the others if ALLOW_NA is on."),
    indicators=[
        dict(code="F1", name="Content Accuracy",
             l1="Teacher makes factual errors that go uncorrected.",
             l2="Mostly accurate but with minor errors or imprecise language.",
             l3="Content is accurate; no errors observed.",
             l4="Content is accurate AND teacher explains WHY (conceptual depth, not just facts).",
             why="FICO V3 (B9) + TEACH + HOTS. 15-25% of teachers in LMICs make content errors."),
        dict(code="F2", name="Use of Academic Language",
             l1="Incorrect or no subject-specific terminology used.",
             l2="Some terms used but not explained or used inconsistently.",
             l3="Key terms used accurately and explained to students.",
             l4="Terms used naturally; students also use them; bilingual bridging (Urdu/English) effective.",
             why="FICO V3 (B10) + OECD + Content Expertise. Academic language is the medium of assessment."),
        dict(code="F3", name="Anticipation of Student Misconceptions",
             l1="Teacher unaware of common misconceptions in this topic.",
             l2="Aware but doesn't address them proactively.",
             l3="Anticipates and addresses at least 1-2 common misconceptions.",
             l4="Systematically surfaces and corrects misconceptions; uses diagnostic questions.",
             why="FICO V3 (B11) + TEACH + L1 diagnosing misconceptions. Shulman's PCK."),
        dict(code="F4", name="Depth of Explanation",
             l1="Superficial/procedural explanation only ('do it this way').",
             l2="Some conceptual explanation but relies on memorization.",
             l3="Explains the 'why' behind procedures; uses multiple representations.",
             l4="Deep conceptual teaching; connects to broader principles; encourages student reasoning.",
             why="HOTS + OECD + TEACH. A teacher who only teaches procedures produces students who only memorize."),
        dict(code="F5", name="Subject-Specific Pedagogy: MATH",
             l1="Math taught purely procedurally; no use of manipulatives or visuals.",
             l2="Some visual aids but conceptual understanding not developed.",
             l3="Uses concrete -> pictorial -> abstract (CPA) progression; manipulatives present.",
             l4="CPA approach mastered; multiple solution strategies explored; math talk norms established.",
             subject="MATH",
             why="FICO V3 (BM12-BM13) + EGMA + Content Expertise: Math. CPA is gold standard for primary math."),
        dict(code="F6", name="Subject-Specific Pedagogy: SCIENCE",
             l1="Science taught from textbook only; no inquiry or observation.",
             l2="Some demonstration but teacher-led; students observe passively.",
             l3="Hands-on activities present; students make predictions and observations.",
             l4="Full inquiry cycle: question -> predict -> investigate -> conclude. Students design investigations.",
             subject="SCIENCE",
             why="FICO V3 (BS12-BS13) + OECD + HOTS. Inquiry-based learning d=0.40."),
        dict(code="F7", name="Subject-Specific Pedagogy: LITERACY / LANGUAGE",
             l1="Reading taught as decoding only; no comprehension strategies.",
             l2="Some reading activities but no explicit strategy instruction.",
             l3="Teacher models reading strategies (prediction, summarizing, questioning). Balanced approach.",
             l4="Balanced literacy: phonics + fluency + vocabulary + comprehension + writing integrated.",
             subject="LITERACY",
             why="FICO V3 (BL12-BL14) + EGRA + Content Expertise: English & Urdu. Balanced literacy wins."),
        dict(code="F8", name="Cross-Curricular Connections",
             l1="Subject taught in complete isolation.",
             l2="Occasional reference to other subjects but not developed.",
             l3="Meaningful connections made to at least one other subject area.",
             l4="Integrated approach; students see how math connects to science connects to language.",
             why="OECD + HOTS + L2-L3 interdisciplinary PBL. Transfer is the ultimate goal of education."),
    ])

FRAMEWORK = {s["code"]: s for s in (SECTION_B, SECTION_C, SECTION_D, SECTION_F)}

FRAMEWORK_DF = pd.DataFrame([
    dict(section=s["code"], section_title=s["title"], code=i["code"], indicator=i["name"],
         L1=i["l1"], L2=i["l2"], L3=i["l3"], L4=i["l4"],
         subject=i.get("subject", ""), module=i.get("module", ""), why=i.get("why", ""))
    for s in FRAMEWORK.values() for i in s["indicators"]
])
ALL_CODES   = FRAMEWORK_DF["code"].tolist()
CODE_ORDER  = {c: k for k, c in enumerate(ALL_CODES)}
CODE2NAME   = dict(zip(FRAMEWORK_DF["code"], FRAMEWORK_DF["indicator"]))
CODE2SECTION= dict(zip(FRAMEWORK_DF["code"], FRAMEWORK_DF["section"]))
SECTION_CODES = {s: FRAMEWORK_DF.loc[FRAMEWORK_DF.section == s, "code"].tolist()
                 for s in FRAMEWORK}

def render_indicator(ind, terse=False):
    """One indicator as a rubric block for the prompt."""
    if terse:
        return (f"{ind['code']} — {ind['name']}\n"
                f"  1={ind['l1']}\n  2={ind['l2']}\n  3={ind['l3']}\n  4={ind['l4']}")
    lines = [f"### {ind['code']} — {ind['name']}"]
    for lvl, key, label in ((1, "l1", "Not Observed / Emerging"), (2, "l2", "Developing"),
                            (3, "l3", "Proficient / Effective"), (4, "l4", "Highly Effective")):
        lines.append(f"  {lvl} ({label}): {ind[key]}")
    if ind.get("subject"):
        lines.append(f"  [Applies only to {ind['subject']} lessons]")
    if INCLUDE_WHY_IN_PROMPT and ind.get("why"):
        lines.append(f"  Context: {ind['why']}")
    return "\n".join(lines)

def render_section_rubric(section_code, codes=None, terse=False):
    s = FRAMEWORK[section_code]
    inds = [i for i in s["indicators"] if codes is None or i["code"] in codes]
    if codes is not None:                            # honour caller's order (shuffling)
        inds = sorted(inds, key=lambda i: codes.index(i["code"]))
    head = f"SECTION {s['code']} — {s['title'].upper()}\nSection guidance: {s['note']}"
    return head + "\n\n" + "\n\n".join(render_indicator(i, terse) for i in inds)


def section_note(code):
    return FRAMEWORK[code]["note"]


N_INDICATORS = len(ALL_CODES)
