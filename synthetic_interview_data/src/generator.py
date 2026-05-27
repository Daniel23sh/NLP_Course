from __future__ import annotations

import random

from src.llm_client import LLMClient
from src.schemas import GeneratedAnswer, ProfileSpec, TARGET_ROLE


QUALITY_WORDS = {
    "poor": "hard to follow and a bit scattered",
    "uneven": "understandable but not very organized",
    "understandable": "mostly understandable",
    "clear": "clear and easy to follow",
    "excellent": "concise, concrete, and well structured",
}


def build_answer_prompt(profile: ProfileSpec, question: dict) -> str:
    is_low_relevance = profile.expected_score_constraints.get("role_relevance", {}).get("max", 5) <= 2
    is_observer = profile.profile_id == "observer_no_personal_task"
    tech = ", ".join(profile.technologies) if profile.technologies else "no specific technical tools"
    if is_low_relevance:
        opening = "Write a natural first-person answer from a student or junior candidate."
        activity_line = f"The answer is about a {question['project_domain']} activity, not software development."
        role_line = "The candidate should not frame the answer around a developer job or technical work."
    else:
        opening = "Write a natural first-person junior developer interview answer."
        activity_line = f"The candidate worked on a {question['project_domain']} project using {tech}."
        role_line = f"The role is {TARGET_ROLE}."
    if is_observer:
        ownership_line = "The candidate did not own a specific task and should describe only watching, listening, attending, or trying to understand."
    else:
        ownership_line = f"Their personal involvement should feel like: {profile.ownership_level}."
    return "\n".join(
        [
            opening,
            role_line,
            f"The interview question is: {question['question']}",
            activity_line,
            f"The project context is {profile.project_type} and the main situation is {profile.challenge_type}.",
            ownership_line,
            f"The explanation should be {QUALITY_WORDS.get(profile.communication_quality, profile.communication_quality)}.",
            f"The technical detail should be {profile.technical_detail}.",
            f"The concrete detail level should be {profile.specificity}.",
            f"The outcome should be {profile.outcome_strength}.",
            _weak_profile_instruction(profile),
            "Return only the candidate answer text.",
            "Do not mention these generation instructions.",
        ]
    )


def _weak_profile_instruction(profile: ProfileSpec) -> str:
    if profile.desired_quality_hint == "high_impact":
        return (
            "Include a concrete project-level result tied to the candidate's own task. Use one believable junior-scale "
            "outcome such as before/after numbers, saved time, reduced manual work, reduced latency, fewer errors, "
            "better load time, improved accuracy, a QA checklist passing, a final demo, or a deliverable used by a "
            "team, user, or client. Do not rely on vague personal learning, 'it helped me', 'it was useful', "
            "'the team moved forward', or 'we improved it' unless the answer explains the observable result. Keep "
            "the scale realistic and avoid exaggerated enterprise claims."
        )
    if profile.profile_group == "official_strong" or profile.desired_quality_hint == "strong_official":
        return "Give a strong but realistic junior answer with concrete personal ownership, implementation details, debugging or design reasoning, clear sequence, tests or verification, and a concrete project result. Keep it natural, not exaggerated."
    if profile.profile_id == "unclear_rambling_answer":
        return "Make the answer naturally rambling, hard to sequence, and vague; do not organize it into a polished situation-action-result story."
    if profile.profile_id == "off_topic_learning_reflection":
        return "Focus on general learning, communication, or study habits rather than a concrete software project; avoid implementation, debugging, or testing details."
    if profile.profile_id == "generic_team_summary_no_personal_role":
        return "Use mostly team-level language; avoid clear candidate-owned actions such as 'I implemented', 'I debugged', or 'I tested'."
    if profile.profile_id == "irrelevant_project_summary":
        return "Describe a barely relevant class activity or general project summary, not a junior software developer project answer."
    if profile.profile_id == "no_outcome_answer":
        return "Do not include a clear result, delivered value, user benefit, measurable outcome, strong learning outcome, usefulness, smooth progress, success, final result, or anything used by someone."
    if profile.profile_id == "vague_tool_mentions_without_explanation":
        return "Mention tools only in passing; avoid implementation details, debugging steps, tradeoffs, algorithms, or technical reasoning."
    if profile.profile_id == "team_only_no_personal_ownership":
        return "Use team-level or group-level language; avoid candidate-owned actions like implementing, building, designing, testing, debugging, fixing, owning, or being responsible for a specific task."
    if profile.profile_id == "observer_no_personal_task":
        return "Use passive observer language: the team or they did the work, while the candidate only watched, listened, attended meetings, or tried to understand. Do not use: I helped, I contributed, my role, I worked on, I organized, I kept track, I followed up, I made sure, I handled, or I supported."
    if profile.profile_id == "rambling_unclear_answer":
        return "Make the answer realistic but rambling; jump between topics, use vague references, and avoid a clean problem-action-result sequence."
    if profile.profile_id == "fragmented_missing_context_answer":
        return "Omit key context, use disconnected details like 'the thing' or 'that part', and leave the project and personal role unclear."
    if profile.profile_id == "severely_unclear_but_realistic_answer":
        return "Make the answer severely unclear but realistic. Use one short messy paragraph made of fragments, vague references, and missing context. Do not explain what the project actually was. Do not write a clean story. Do not write polished paragraphs. Avoid sequence markers such as first, then, after, finally, or overall. Use phrases like the thing, that part, some stuff, I do not really remember, and it kind of changed."
    if profile.profile_id == "unfinished_project_no_result":
        return "Describe work that stayed incomplete, untested, or unused; forbid usefulness, strong learning, smooth progress, success, final results, delivery, deployment, user value, and completed outcomes."
    if profile.profile_id == "irrelevant_non_software_project":
        return "Describe a non-software class activity with weak connection to junior developer work; avoid implementation, debugging, testing, API, database, frontend, backend, or model-training evidence."
    if profile.profile_id in {
        "non_software_team_activity",
        "generic_school_assignment_no_coding",
        "project_manager_observer_role",
        "presentation_only_project",
        "learning_reflection_without_project",
        "design_only_no_implementation",
        "irrelevant_volunteer_project",
    }:
        return "Forbid coding, debugging, testing, implementation, technical decisions, project tools, task boards, deadlines, junior developer framing, software job framing, API/database/frontend/backend/deployment/model references, and strong project narrative. Keep it clearly about non-software teamwork, school reflection, presentation-only work, volunteer organization, planning without implementation, or observing rather than building."
    if profile.profile_id == "vague_tools_no_explanation":
        return "Mention at most one or two tools vaguely. Avoid architecture, tradeoffs, debugging, algorithms, data flow, API details, database details, testing, performance reasoning, or specific code changes."
    if profile.profile_id == "polished_but_shallow_answer":
        return "Make the answer polished and easy to follow, but keep it shallow: general statements, almost no implementation detail, and no concrete project outcome."
    return "Keep the answer natural and junior-realistic."


def scenario_family(question: dict, profile: ProfileSpec) -> str:
    return "|".join(
        [
            question.get("question_type", "unknown_question"),
            question.get("project_domain", "unknown_domain"),
            profile.profile_id,
            profile.challenge_type,
            profile.desired_quality_hint,
        ]
    )


def _ownership_sentence(profile: ProfileSpec) -> str:
    if profile.ownership_level == "hidden_team_only":
        return "Our team worked on it together, and I mostly followed the general task list without a separate part that is easy to describe."
    if profile.ownership_level == "assisted":
        return "I helped with a small part of the work after a teammate showed me where to start."
    if profile.ownership_level == "personally_owned":
        return "I personally owned the main implementation task: I implemented the change, I tested it, I debugged the edge cases, and I shared the result for review."
    return "I handled a clear piece of the implementation and checked it with my teammate before we merged it."


def _technical_sentence(profile: ProfileSpec, domain: str) -> str:
    tech = ", ".join(profile.technologies[:3]) if profile.technologies else "the project tools"
    if profile.technical_detail == "absent":
        return "I do not remember many technical details beyond trying to make the project work."
    if profile.technical_detail == "low":
        return f"I used {tech}, but my explanation is mostly about the feature rather than the implementation."
    if profile.technical_detail == "very_high":
        return f"I used {tech}, reproduced the issue locally, inspected logs and inputs, compared two implementation options, and tested edge cases before choosing the simpler fix."
    if profile.technical_detail == "high":
        return f"I used {tech}, checked the failing behavior, changed the code, and tested the result with normal and edge-case inputs."
    return f"I used {tech} and made a basic implementation change in the {domain}."


def _problem_sentence(profile: ProfileSpec) -> str:
    if profile.challenge_type in {"debugging", "performance", "design_decision"}:
        return "The challenge was that the first approach did not behave correctly, so I broke it into smaller checks, tested one assumption at a time, and adjusted the solution after each result."
    if profile.challenge_type == "failure_learning":
        return "At first the project did not work as expected, and I learned to check the assumptions earlier instead of changing several things at once."
    if profile.challenge_type == "overview":
        return "There was not one major technical obstacle that I can describe clearly."
    return "I had to coordinate the task with teammates and make sure my part still fit the shared project."


def _impact_sentence(profile: ProfileSpec) -> str:
    if profile.desired_quality_hint == "high_impact":
        return "The result was concrete: the change was used in the final demo and reduced repeated manual checking from about 30 minutes to 5 minutes."
    if profile.outcome_strength == "none":
        return "I do not remember a clear final result from that task."
    if profile.outcome_strength == "vague":
        return "It helped the project move forward, although I cannot point to a specific measured result."
    if profile.outcome_strength == "measurable":
        return "After the change, the page loaded about 30 percent faster, users saw fewer errors, and the team could deliver the milestone."
    if profile.outcome_strength == "clear_project_value":
        return "After the change, the feature was easier to use, the test flow was more stable, and the team could finish the milestone."
    return "The main result was that I learned how to explain the tradeoff and verify the fix more carefully."


def _clarity_order(sentences: list[str], profile: ProfileSpec, rng: random.Random) -> list[str]:
    if profile.communication_quality == "poor":
        shuffled = list(sentences)
        rng.shuffle(shuffled)
        return shuffled
    if profile.communication_quality == "uneven":
        return [sentences[0], sentences[2], sentences[1], *sentences[3:]]
    return sentences


def _mock_answer(profile: ProfileSpec, question: dict, seed: int) -> str:
    rng = random.Random(seed)
    if profile.profile_id == "unclear_rambling_answer":
        return (
            "There was this project and I was around some parts of it, and at first it was kind of about a script "
            "but also some screens and then we changed what we were doing. I helped here and there, mostly trying "
            "to understand what was needed, and sometimes I looked at pieces that were confusing. It is hard to "
            "explain the order because different things happened at the same time, but I remember trying to keep "
            "up and asking questions when something did not make sense."
        )
    if profile.profile_id == "off_topic_learning_reflection":
        return (
            "During my studies I learned that communication and consistency matter a lot. I was not describing one "
            "specific software project here, but I noticed that when people share updates and stay organized the "
            "group works better. It helped me understand teamwork and how to be more responsible in class."
        )
    if profile.profile_id == "generic_team_summary_no_personal_role":
        return (
            "Our team worked on a dashboard project together. We discussed the data, divided general areas, and "
            "helped each other when something was confusing. The group made progress over time, but my exact part "
            "was not very separate from what everyone else was doing."
        )
    if profile.profile_id == "irrelevant_project_summary":
        return (
            "One class activity I remember was preparing a short presentation about how teams organize their work. "
            "It was more about planning and communication than building software. The main thing was summarizing "
            "ideas clearly and making sure the group presentation made sense."
        )
    if profile.profile_id == "no_outcome_answer":
        return (
            "I worked on a small project task where I helped with a few pieces of the interface and checked some "
            "basic behavior. I remember spending time trying to understand the files and making a small change, but "
            "I do not remember whether it was used, finished, or useful in the final project."
        )
    if profile.profile_id == "vague_tool_mentions_without_explanation":
        return (
            "I worked on a project where we used React and Python with an API. I helped with some parts and saw how "
            "the tools connected, but I do not remember the implementation details or the reasoning behind the code. "
            "It was mostly a chance to get familiar with the stack."
        )
    if profile.profile_id == "team_only_no_personal_ownership":
        return (
            "The group had a dashboard assignment where different people talked about the data and the screens. "
            "The work was shared in a general way, and the group mostly followed the checklist from class. "
            "There was not a separate personal task described, just participation in the team process."
        )
    if profile.profile_id == "observer_no_personal_task":
        return (
            "The team discussed what needed to happen and they divided the work among themselves. I mostly listened "
            "to the discussion, followed along with what others were deciding, and learned from seeing how the group "
            "organized the task. There was no specific personal task or owned part to describe."
        )
    if profile.profile_id == "rambling_unclear_answer":
        return (
            "There was this thing from class and then the page or maybe the script part was confusing, and the order "
            "is hard to explain. Some pieces happened before other pieces but I am not sure which one was first. "
            "People were changing parts, then we looked at something else, and the issue was kind of still there, "
            "so the explanation jumps around a bit."
        )
    if profile.profile_id == "fragmented_missing_context_answer":
        return (
            "That part was connected to the thing we had to submit. The issue was in one place, then another part "
            "had to be changed, but the setup is hard to explain. Some files were involved and the output was not "
            "really clear. I remember looking at it, but not enough context is here to show what the project was."
        )
    if profile.profile_id == "severely_unclear_but_realistic_answer":
        return (
            "There was the thing from that class, some part, I do not really remember what the main project was, "
            "people changed stuff and that part was confusing, maybe it worked or maybe it was just submitted, "
            "the order is hard to explain, not enough context, I was around some of it but the task is not clear."
        )
    if profile.profile_id == "unfinished_project_no_result":
        return (
            "I worked on an integration idea for a small portfolio task. I explored the files and tried a basic "
            "connection, but it stayed incomplete and was not used by anyone. I did not finish testing it, and I "
            "cannot point to a final result from that work."
        )
    if profile.profile_id == "irrelevant_non_software_project":
        return (
            "One class activity I remember was preparing a short presentation about how students organize group work. "
            "It was mostly about planning, communication, and dividing slides. It was not a software project and did "
            "not involve writing or testing code."
        )
    if profile.profile_id == "non_software_team_activity":
        return (
            "The activity was a group exercise about planning a class event. People talked about timing, materials, "
            "and who would present each part. It was useful for teamwork, but it was not about software, coding, or "
            "technical problem solving."
        )
    if profile.profile_id == "generic_school_assignment_no_coding":
        return (
            "One assignment was a short written summary for a general course. The main work was reading, discussing "
            "the topic, and turning in a simple response. There was no coding, implementation, testing, or technical "
            "decision involved."
        )
    if profile.profile_id == "project_manager_observer_role":
        return (
            "The group had a planning task where one person kept track of dates and others discussed what should be "
            "done. I mostly observed the planning conversation and learned how people organized responsibilities. It "
            "did not include building or testing software."
        )
    if profile.profile_id == "presentation_only_project":
        return (
            "The project was a presentation for class. The work was mostly choosing points for slides, practicing the "
            "talk, and making sure the presentation order made sense. It did not involve coding or implementing a "
            "technical feature."
        )
    if profile.profile_id == "learning_reflection_without_project":
        return (
            "I learned that communication matters when people work together. This was more of a reflection from class "
            "than a specific project. It was not a software project, and I am not describing debugging, testing, or a "
            "technical decision."
        )
    if profile.profile_id == "design_only_no_implementation":
        return (
            "The work was a design mockup for an idea. It stayed at the planning and layout discussion stage, without "
            "implementation. There was no frontend, backend, database, deployment, or testing work to explain."
        )
    if profile.profile_id == "irrelevant_volunteer_project":
        return (
            "I helped with organizing a volunteer event where people planned tables, timing, and communication. It was "
            "a useful teamwork experience, but it was not related to software development or technical project work."
        )
    if profile.profile_id == "vague_tools_no_explanation":
        return (
            "I used some tools like React and Firebase in a small app idea. I mostly followed examples and connected "
            "things in a basic way, but I cannot explain the architecture, data flow, or why the tools were chosen. "
            "The answer is more about naming the stack than explaining the implementation."
        )
    if profile.profile_id == "polished_but_shallow_answer":
        return (
            "I worked on a small React project for my portfolio. The goal was to make a simple interface and practice "
            "organizing a project. I kept the answer easy to explain: I set up the page, made it look consistent, and "
            "learned how to present my work. I do not have many implementation details or a concrete outcome to share."
        )
    if profile.profile_id == "deployed_feature_with_usage":
        return (
            "I worked on a backend API feature for a course project where users needed to save form drafts. "
            "I added the save endpoint, checked the SQLite write path, and tested the empty-field cases. "
            "The feature was used in our final demo, and teammates could complete the demo flow without losing draft data."
        )
    if profile.profile_id == "measurable_performance_improvement":
        return (
            "I handled a slow query in a backend API project. I compared the original SQL with a filtered query, added "
            "a small index, and tested the same requests before and after the change. Average response time went from "
            "1.8 seconds to about 700 ms on our test dataset, which made the demo feel much smoother."
        )
    if profile.profile_id == "reduced_manual_work":
        return (
            "I wrote a Python automation script for a weekly CSV report. Before that, a teammate copied values into "
            "the spreadsheet by hand, which took around 30 minutes. After my script parsed the file and generated the "
            "summary, the same report took about 5 minutes and the team used it for the final submission."
        )
    if profile.profile_id == "accuracy_or_quality_improvement":
        return (
            "In an NLP course project, I changed the text preprocessing and compared validation results across two "
            "runs. I removed noisy tokens, fixed inconsistent labels in the split, and reran the evaluation. Validation "
            "accuracy increased from 62% to 76%, so we used that version in the final demo."
        )
    if profile.profile_id == "successful_demo_or_client_value":
        return (
            "I worked on a dashboard filter for a class client-style demo. I connected the date and category filters "
            "to the data table, tested a few empty-result cases, and fixed the display text. In the final demo, the "
            "reviewer used the filter to narrow the report quickly instead of scrolling through the full dataset."
        )
    if profile.profile_id == "qa_or_error_reduction_fix":
        return (
            "I fixed an upload bug in a small testing-framework project. I reproduced the failing image case, added a "
            "validation check, and reran the QA checklist on the tested devices. The checklist went from 18 errors to "
            "3, and the remaining items were unrelated styling issues."
        )
    if profile.profile_id == "before_after_metric_improvement":
        return (
            "I worked on a CLI data-processing utility where the slow part was reading the same file multiple times. "
            "I changed it to load the data once, tested the output against the old version, and measured the run time. "
            "The command went from about 42 seconds to 16 seconds for the sample folder."
        )
    sentences = [
        f"In a {question['project_domain']} project, I worked on an example related to this question.",
        _ownership_sentence(profile),
        _technical_sentence(profile, question["project_domain"]),
        _problem_sentence(profile),
        _impact_sentence(profile),
        "It was useful interview experience because it helped me connect a project story to junior developer work.",
    ]
    if profile.specificity in {"concrete", "highly_specific"}:
        sentences.append("A detail I remember is writing down the failing case before changing the code so I could compare the result afterward.")
    if profile.specificity == "generic":
        sentences = [sentence.replace("API", "project").replace("logs", "things") for sentence in sentences[:5]]
    return " ".join(_clarity_order(sentences, profile, rng))


def generate_answer(profile: ProfileSpec, question: dict, mode: str, seed: int, model: str) -> GeneratedAnswer:
    if mode == "mock":
        answer = _mock_answer(profile, question, seed)
    else:
        answer = LLMClient(mode="openai", model=model).generate(build_answer_prompt(profile, question), seed=seed)
    return GeneratedAnswer(
        example_id=question["example_id"],
        question_id=question["question_id"],
        question_type=question["question_type"],
        project_domain=question["project_domain"],
        question=question["question"],
        profile=profile,
        answer=answer.strip(),
        scenario_family=question.get("scenario_family") or scenario_family(question, profile),
        metadata={"generation_backend": mode, "model": model, "seed": seed},
    )
