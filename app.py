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
   - Keep these brief, but present.

3. Weave in Small, Parenthetical Asides:
   - Short side remarks for self-deprecating humor or tangential points.

4. Organize with Heading + Rich Detail:
   - For each heading line, use exactly `# ` (no `##` or `###`).
   - Under each heading, you can provide multiple paragraphs, bullet lists, enumerations, or other expansions for sub-points. 
   - The result should be comprehensive, not just 1-3 lines.

5. Collegial, Pragmatic Optimism:
   - Show hope for solutions while recognizing difficulties.

6. Use Examples and References:
   - Keep them concise but illustrative.

7. Combine Technical Depth + Accessibility:
   - Enough detail for specialists, but explain jargon as needed.

8. Mild Wit/Personal Touch is Fine:
   - Substance first, but a bit of warmth or playful phrasing is welcome.

9. Engage Opposing/Uncertain Views:
   - Note how you’d address or monitor potential downsides.

10. Minimal “LLM-y” Output:
   - Just a plain outline with `#` headings and rich sub-points in paragraphs or bullet points.

Remember: Each major heading line MUST start with `# ` and no further hash marks. Under it, you can have all the detail you want (paragraphs, bullet points, etc.).
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
Generate a comprehensive outline for a long-form blog post using ONLY top-level headings, each line starting with '# ' (a single hash and a space).

Under each heading:
- Provide multiple paragraphs, bullet lists, enumerations, or detailed expansions for subpoints (but do NOT use additional '#' symbols).
- Aim for a thorough, multi-section structure (~5-9 headings).
- The style_guide indicates we want a robust, detailed approach, so feel free to add depth and sub-points as bullets or paragraphs.

Style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Output only the outline text. No disclaimers, code fences, or extra commentary.
- Each heading line = '# Some Title'
- Sub-points or expansions go under that heading as normal text, bullets, paragraphs, etc.
"""

OUTLINE_USER = """Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Please produce a comprehensive outline with multiple major headings (5-9). Each heading line must begin with "# ".
Below each heading, include rich detail: multiple paragraphs, bullet lists, enumerations, etc.
No disclaimers or code fences, just the raw outline text.
"""

OUTLINE_FEEDBACK_SYSTEM = f"""
You are an outline revision assistant. 
You will update the existing outline given user feedback. 
Outline must keep the same top-level heading format: each heading line starts with '# ', and no sub-headers with '##'.

Style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Keep or modify headings as needed, but do not introduce secondary '#' lines.
- Retain or expand the bullet points, paragraphs, etc. to remain comprehensive.
- Output only the updated outline, with no disclaimers or code fences.
"""

OUTLINE_FEEDBACK_USER = """ORIGINAL OUTLINE:
{original_outline}

USER FEEDBACK:
{feedback}

Revise the outline accordingly. Each heading starts with "# ", and the detail can be multiple paragraphs or bullets. Output only the updated outline text.
"""

BLOG_SECTION_SYSTEM = f"""
You are a blogging assistant writing a blog piece in multiple sections (one for each top-level heading). 
You'll output raw HTML for each section.

Style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- You have a partial blog so far in HTML. 
- You must write the next section in raw HTML (no disclaimers, no code fences).
- Because each heading is top-level only, each heading chunk from the outline is its own section.
- Add enough substance for each section so the final blog is around 2000 words total.
"""

BLOG_SECTION_USER = """Partial blog so far:
{partial_blog}

Current heading + sub-points from outline:
{section_heading}

Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Write this section in HTML, continuing the blog. 
Output only this section's HTML (no disclaimers or code fences).
"""

FEEDBACK_SYSTEM = f"""
You are a blogging assistant updating the final assembled blog post based on user feedback. 
Keep the structure unless feedback requests changes.

Style_guide:
{style_guide}

Instructive excerpt:
{instructive_excerpt}

INSTRUCTIONS:
- Output only the updated blog in raw HTML. No disclaimers, code fences, or other commentary.
"""

FEEDBACK_USER = """ORIGINAL BLOG:
{original_blog}

USER FEEDBACK:
{feedback}

Update the blog accordingly, preserving the existing HTML structure unless changes are requested. Output only the updated HTML, nothing else.
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
    We only want headings that start with "# ".
    We'll treat each heading and its subsequent lines as a single "section block"
    until we hit the next heading line starting with "# ".
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
        if line.strip().startswith("# "):
            # If there's an existing heading, store it
            if current_heading or current_content:
                sections.append(add_section(current_heading, current_content))
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    # last chunk
    if current_heading or current_content:
        sections.append(add_section(current_heading, current_content))

    if not sections:
        # fallback: entire outline
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
    core_thesis = st.text_area("Core Argument/Thesis:", height=100)

    st.subheader("2. Enter Supplementary Materials")
    supplementary_materials = st.text_area("Supplementary Materials:", height=100)

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
        st.subheader("3. Current Outline (Top-Level Headings + Rich Detail)")
        st.text_area("Generated Outline:", value=st.session_state["generated_outline"], height=350, key="current_outline_display")

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

        # parse outline into sections
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

                    # Show user the new section
                    section_title = section_heading.split("# ",1)[-1]
                    st.markdown(f"**Section {i+1}**: {section_title}")
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
