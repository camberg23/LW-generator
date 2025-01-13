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
   - “We suspect…,” “We’re still exploring…,” “We might be missing something here,” or “Others may disagree.”  
   - Keep these brief.

3. Weave in Small, Parenthetical Asides:
   - Short side remarks to express self-deprecating humor or tangential points.

4. Organize with Section Headers:
   - For the outline, only use single `#` headings. Do NOT create subheadings with `##`.

5. Be Collegial, Pragmatic, Lightly Optimistic:
   - Show hope or positivity while recognizing difficulty.

6. Include Compact Examples and References:
   - Short real-world/historical references.

7. Technical Depth + Accessibility:
   - Enough detail for specialists, but define jargon for general readers.

8. Occasional Mild Wit or Personal Touch:
   - A bit of playful phrasing is fine.

9. Engage Opposing/Uncertain Views:
   - Note how you’d address or monitor potential downsides.

10. Minimal “LLM-y” Output:
   - Mostly plain text with top-level headings for the outline. For the blog, output raw HTML section by section.
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
Generate an outline for a blog post with ONLY top-level headings, each prefixed by a single '#' followed by a space. 
After each heading, include 1-3 sentences describing what's in that section.

For example:
# Introduction
Brief text describing the intro

# Key Arguments
One or two lines explaining the focus

And so on...

Follow the style rules from the style_guide below:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Only create ~5-7 major headings. No subheadings with '##' or '###'.
- After each heading, provide a short descriptive paragraph.
- Output only the outline text. No disclaimers, no code fences, no extra commentary.
"""

OUTLINE_USER = """Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Please create an outline with ~5-7 major headings. Each heading line starts with "# " (no subheadings).
Under each heading, write a short descriptive paragraph (1-3 sentences). 
No disclaimers or code fences. Keep it minimal.
"""

OUTLINE_FEEDBACK_SYSTEM = f"""
You are an outline revision assistant. 
You will update the existing outline given user feedback. 
The outline must keep the same format: Only lines starting with "# " for headings, followed by 1-3 lines of descriptive text.

Style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Retain or modify the existing headings and text as needed.
- No disclaimers or extra commentary. Output just the updated outline text.
"""

OUTLINE_FEEDBACK_USER = """ORIGINAL OUTLINE:
{original_outline}

USER FEEDBACK:
{feedback}

Update the outline accordingly, preserving the single '#' heading style. Output only the updated outline.
"""

BLOG_SECTION_SYSTEM = f"""
You are a blogging assistant writing a blog piece in multiple sections, one section at a time in raw HTML.

Style guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- You have a partial blog so far in HTML. You must add the next section in raw HTML.
- Each top-level heading from the outline is processed in order. 
- Output only the new section in HTML, no disclaimers or code fences, no reprinting the old sections.
- The total blog should be around 2000 words, so each section is substantive.
"""

BLOG_SECTION_USER = """Partial blog so far:
{partial_blog}

Current heading to write (with its short description):
{section_heading}

Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Write this entire section in HTML, continuing the blog. Only output this section's HTML, no disclaimers or code fences.
"""

FEEDBACK_SYSTEM = f"""
You are a blogging assistant updating the final assembled blog post based on user feedback. 
Maintain original structure unless feedback requests changes.

Style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Output only the updated blog in raw HTML. No disclaimers, code fences, or extra commentary.
"""

FEEDBACK_USER = """ORIGINAL BLOG:
{original_blog}

USER FEEDBACK:
{feedback}

Update the blog accordingly, preserving the existing HTML structure. Output only the updated HTML, nothing else.
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
    We only want headings that start with "# " exactly.
    We treat the heading line + subsequent lines as one "section block"
    until the next heading line is found.
    """
    lines = outline_text.splitlines()
    sections = []
    current_heading = ""
    current_content = []

    def add_section(heading, content):
        # Combine heading & its descriptive lines
        heading_plus_content = heading
        if content:
            heading_plus_content += "\n" + "\n".join(content)
        return heading_plus_content.strip()

    for line in lines:
        if line.strip().startswith("# "):
            # If there's an existing heading, store it
            if current_heading or current_content:
                sections.append(add_section(current_heading, current_content))
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    # Last chunk
    if current_heading or current_content:
        sections.append(add_section(current_heading, current_content))

    if not sections:
        # fallback to entire text
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
        st.subheader("3. Current Outline (Top-Level Headings Only)")
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
        st.subheader("4. Generate the Blog (One Heading at a Time)")

        st.session_state["outline_sections"] = parse_outline_into_sections(st.session_state["generated_outline"])

        if st.button("Generate All Sections in Order"):
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

                    st.markdown(f"**Section {i+1}**: {section_heading.split('# ',1)[-1]}")
                    st.html(section_html)
                    st.write("---")
                    time.sleep(1)

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
