# alignment_blog_app.py

import streamlit as st
import re
import anthropic
import time
from openai import OpenAI

########################################
# MODEL HANDLING
########################################

def is_o1_model(m):
    return m.startswith("o1-")

def is_gpt4o_model(m):
    return m == "gpt-4o"

def is_anthropic_model(m):
    return m.startswith("anthropic")

def adjust_messages_for_o1(messages):
    adjusted = []
    for msg in messages:
        if msg["role"] == "system":
            content = "INSTRUCTIONS:\n" + msg["content"]
            adjusted.append({"role": "user", "content": content})
        else:
            adjusted.append(msg)
    return adjusted

########################################
# GLOBAL CLIENTS
########################################

openai_client = None
anthropic_client = None

########################################
# STYLE GUIDE AND EXCERPTS
########################################

style_guide = """
When producing text, follow these instructions to achieve a voice and tone reminiscent of thoughtful long-form posts in alignment/rationalist communities:

1. Establish Rapport and a Communal Voice:
   - Speak as part of a group exploring ideas collaboratively. Use inclusive pronouns (“we,” “our,” “let’s consider…”).
   - Acknowledge doubts or open questions without undermining your own authority.

2. Use Subtle Disclaimers and Qualifiers:
   - Incorporate phrases like “We suspect…,” “We’re still exploring…,” “We might be missing something here,” or “Others may disagree.”  
   - Keep disclaimers brief, so they don’t derail the main argument.

3. Weave in Small, Parenthetical Asides:
   - Add short remarks in parentheses to express self-deprecating humor or highlight tangential points.
   - Avoid overusing these—sprinkle them organically.

4. Organize with Section Headers and Occasional Bullets:
   - Use short, clear headings (e.g. # Introduction, # Main Argument, ## Key Idea, etc.).
   - If needed, subheadings can be done with ## or ###.

5. Collegial, Pragmatic Optimism:
   - Show hope for solutions while recognizing difficulties.
   - Avoid overhype. Convey that solutions need real diligence.

6. Compact Examples and References:
   - Short real-world/historical examples. Keep them concise.

7. Technical Depth + Accessibility:
   - Enough detail for specialists, but define or paraphrase jargon for lay readers.
   - Introduce major formalisms carefully.

8. Occasional Mild Wit or Personal Touch:
   - Gently playful phrasing is fine, but keep substance first.

9. Genuine Engagement with Opposing/Uncertain Views:
   - Show how you’d address or monitor potential downsides.

10. Minimal “LLM-y” Outputs:
   - Don’t overuse special formatting. This is mostly prose with headings.

Remember: HEADINGS for the outline must start with # or ## (or ### if needed). 
"""

instructive_excerpt = """
Excerpt Demonstrating the Tone:
\"We've been laying the groundwork for alignment policy in a Republican-controlled government. 
Self-deprecating note: we're still learning and iterating on these specific approaches, being fairly new to the policy space ourselves. 
What follows isn't meant to be a complete solution, but rather some early strategies that have worked better than we initially expected in engaging conservative policymakers on AI safety.

We suspect that some might disagree with this angle, but from our vantage point, acknowledging ideological differences up front while providing strong technical foundations often leads to more common ground than one would predict.\"
"""

########################################
# PROMPTS
########################################

OUTLINE_SYSTEM = f"""
You are a content planning assistant. 
Generate an outline for a long-form blog post in a strict format using # or ## at the start of each heading:

Example:
# Introduction
Intro text

## Subpoint
Text for subpoint

# Next Major Heading
More text here

Follow the style rules from the style_guide below:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Use # for major headings and ## for subheadings.
- After a heading line, include a few lines of descriptive text for that section.
- Output only the outline text, no disclaimers or code fences.
- Format example:

# Introduction
Some opening text
## Key Topic
Details, bullets, disclaimers, etc.

No other commentary. End of instructions.
"""

OUTLINE_USER = """Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Generate an outline (headings, bullet points, etc.) that will serve as the skeleton for a long-form blog post. 
Ensure it is cohesive, addresses the main argument, and integrates the supplementary materials where appropriate.
Remember to begin headings with '#' or '##' exactly, and do not wrap output in backticks.
"""

OUTLINE_FEEDBACK_SYSTEM = f"""
You are an outline revision assistant. 
You will update the existing outline given user feedback. 
Output must remain in the same #/## heading format, with no disclaimers or code fences.

Refer to style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Retain or modify the existing # and ## headings as appropriate.
- Integrate user feedback as requested. 
- Output only the updated outline text, no other disclaimers.
- Example format:

# Introduction
some lines about introduction
## subheading
more lines
"""

OUTLINE_FEEDBACK_USER = """ORIGINAL OUTLINE:
{original_outline}

USER FEEDBACK:
{feedback}

Update the outline accordingly, preserving the #/## heading structure. Output only the new outline text.
"""

BLOG_SECTION_SYSTEM = f"""
You are a blogging assistant writing a blog piece in multiple sections. 
You have a partial blog so far, and must add the next section in raw HTML.

Follow the style guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Incorporate the main argument and supplementary materials if relevant.
- Output only the HTML for the new section. No disclaimers, code fences, or extra commentary.
- Aim for a total blog near 2000 words, so each section should be substantive.
- Maintain continuity with the partial blog so far, do NOT rewrite or restate previous sections.
"""

BLOG_SECTION_USER = """Partial blog so far:
{partial_blog}

Next section heading (and text) from outline:
{section_heading}

Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Write only the new section in HTML, continuing the blog. Output HTML only, no disclaimers or code fences.
"""

FEEDBACK_SYSTEM = f"""
You are a blogging assistant updating a blog post based on user feedback. 
Maintain the original structure and text as much as possible unless the feedback requests changes. 

Follow the style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Output only the updated blog post in raw HTML. No disclaimers, code fences, or extraneous commentary.
"""

FEEDBACK_USER = """ORIGINAL BLOG:
{original_blog}

USER FEEDBACK:
{feedback}

Update the blog accordingly, preserving format as HTML. Output only updated HTML, nothing else.
"""

########################################
# COMPLETION FUNCTION
########################################

def run_completion(messages, model, verbose=False):
    """Handles chat completions for O1, GPT-4o, or Anthropic models."""
    if is_o1_model(model):
        adjusted = adjust_messages_for_o1(messages)
        if verbose:
            st.write("**Using O1 Model:**", model)
            st.write("**Messages:**", adjusted)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=adjusted
        )
        return completion.choices[0].message.content

    elif is_gpt4o_model(model):
        if verbose:
            st.write("**Using GPT-4o Model:**", model)
            st.write("**Messages:**", messages)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return completion.choices[0].message.content

    elif is_anthropic_model(model):
        if verbose:
            st.write("**Using Anthropic Model:**", model)
            st.write("**Messages:**", messages)
        system_str_list = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] != "system"]
        system_str = "\n".join(system_str_list) if system_str_list else None

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 8192,
            "messages": user_assistant_msgs
        }
        if system_str:
            kwargs["system"] = system_str

        response = anthropic_client.messages.create(**kwargs)
        return response.content[0].text.strip()

    else:
        return f"Unsupported model: {model}"

########################################
# OUTLINE PARSING
########################################

def parse_outline_into_sections(outline_text):
    """
    We treat lines starting with '#' or '##' as new headings.
    The heading line + subsequent lines (until the next heading) become a single "section block."

    Example:
    # Introduction
    text
    ## subheading
    text
    # Next heading
    text
    """
    lines = outline_text.splitlines()
    sections = []
    current_heading = ""
    current_content = []

    def add_section(heading, content):
        heading_plus_content = heading
        if content:
            heading_plus_content += "\n" + "\n".join(content)
        return heading_plus_content.strip()

    for line in lines:
        if line.strip().startswith("#"):
            # we've hit a new heading; store the previous if it exists
            if current_heading or current_content:
                sections.append(add_section(current_heading, current_content))
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    # last chunk
    if current_heading or current_content:
        sections.append(add_section(current_heading, current_content))

    # fallback if none found
    if not sections:
        sections = [outline_text]
    return sections

########################################
# STREAMLIT APP
########################################

def main():
    global openai_client, anthropic_client

    st.title("AI Alignment Blog Generator (Section-by-Section)")

    # Initialize clients
    openai_client = OpenAI(api_key=st.secrets['API_KEY'])
    anthropic_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])

    # LLM selection
    llm_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]
    outline_model = st.selectbox("Outline Model:", llm_options, index=0)
    outline_feedback_model = st.selectbox("Outline Feedback Model:", llm_options, index=0)
    section_model = st.selectbox("Section-by-section Blog Model:", llm_options, index=0)
    feedback_model = st.selectbox("Final Blog Feedback Model:", llm_options, index=0)

    if "generated_outline" not in st.session_state:
        st.session_state["generated_outline"] = ""
    if "outline_sections" not in st.session_state:
        st.session_state["outline_sections"] = []
    if "blog_sections" not in st.session_state:
        st.session_state["blog_sections"] = []
    if "generated_blog" not in st.session_state:
        st.session_state["generated_blog"] = ""

    st.subheader("1. Enter Core Argument/Thesis")
    core_thesis = st.text_area("Core Argument/Thesis:", height=120)

    st.subheader("2. Enter Supplementary Materials")
    supplementary_materials = st.text_area("Supplementary Materials:", height=120)

    st.write("Both fields above must be filled before generating an outline.")

    # Generate Outline
    if st.button("Generate Outline"):
        if not core_thesis.strip() or not supplementary_materials.strip():
            st.warning("Please provide both the core argument/thesis and supplementary materials.")
        else:
            with st.spinner("Generating Outline..."):
                messages = [
                    {"role": "system", "content": OUTLINE_SYSTEM},
                    {"role": "user", "content": OUTLINE_USER.format(
                        core_thesis=core_thesis,
                        supplementary_materials=supplementary_materials
                    )}
                ]
                outline = run_completion(messages, outline_model)
                st.session_state["generated_outline"] = outline
                st.session_state["blog_sections"] = []
                st.session_state["generated_blog"] = ""

    # Show Outline + Outline Feedback
    if st.session_state["generated_outline"]:
        st.subheader("3. Current Outline (Structured with # and ##)")
        st.text_area("Generated Outline:", value=st.session_state["generated_outline"], height=300, key="current_outline_display")

        # Let user provide feedback on Outline
        outline_feedback = st.text_area("Feedback on Outline (optional):", height=80)

        if st.button("Incorporate Outline Feedback"):
            if not outline_feedback.strip():
                st.warning("Please enter some feedback to incorporate.")
            else:
                with st.spinner("Updating Outline..."):
                    messages = [
                        {"role": "system", "content": OUTLINE_FEEDBACK_SYSTEM},
                        {"role": "user", "content": OUTLINE_FEEDBACK_USER.format(
                            original_outline=st.session_state["generated_outline"],
                            feedback=outline_feedback
                        )}
                    ]
                    updated_outline = run_completion(messages, outline_feedback_model)
                    st.session_state["generated_outline"] = updated_outline
                    st.session_state["blog_sections"] = []
                    st.session_state["generated_blog"] = ""
                st.success("Outline updated with feedback!")
                st.rerun()

    # Generate blog sections
    if st.session_state["generated_outline"]:
        st.subheader("4. Generate the Blog (Section by Section)")

        # parse outline into sections
        st.session_state["outline_sections"] = parse_outline_into_sections(st.session_state["generated_outline"])

        if st.button("Generate All Sections"):
            st.session_state["blog_sections"] = []
            partial_blog = ""
            with st.spinner("Generating all sections..."):
                for i, section_heading in enumerate(st.session_state["outline_sections"]):
                    messages = [
                        {"role": "system", "content": BLOG_SECTION_SYSTEM},
                        {"role": "user", "content": BLOG_SECTION_USER.format(
                            partial_blog=partial_blog,
                            section_heading=section_heading,
                            core_thesis=core_thesis,
                            supplementary_materials=supplementary_materials
                        )}
                    ]
                    section_html = run_completion(messages, section_model)
                    st.session_state["blog_sections"].append(section_html)
                    partial_blog += section_html

                    st.markdown(f"**Section {i+1}**:")
                    st.html(section_html)
                    st.write("---")

            st.session_state["generated_blog"] = partial_blog

    # Display final blog + feedback
    if st.session_state["generated_blog"]:
        st.subheader("5. Review or Edit the Completed Blog")
        st.markdown("Below is the entire assembled HTML. You can copy it or download it.")

        st.html(st.session_state["generated_blog"])

        st.download_button(
            label="Download Blog as HTML",
            data=st.session_state["generated_blog"].encode("utf-8"),
            file_name="generated_blog.html",
            mime="text/html"
        )

        # Feedback / Update
        st.write("You can provide feedback and have the blog updated automatically below.")
        feedback_text = st.text_area("Feedback on Blog:", height=100)
        if st.button("Incorporate Blog Feedback"):
            if not feedback_text.strip():
                st.warning("Please enter some feedback to incorporate.")
            else:
                with st.spinner("Incorporating Feedback..."):
                    messages = [
                        {"role": "system", "content": FEEDBACK_SYSTEM},
                        {"role": "user", "content": FEEDBACK_USER.format(
                            original_blog=st.session_state["generated_blog"],
                            feedback=feedback_text
                        )}
                    ]
                    updated_blog = run_completion(messages, feedback_model)
                    st.session_state["generated_blog"] = updated_blog
                st.success("Feedback Incorporated! See updated blog below.")
                st.rerun()

if __name__ == "__main__":
    main()
