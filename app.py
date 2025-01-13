# alignment_blog_app.py

import streamlit as st
import re
import anthropic
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
            # Convert system instructions into user instructions
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

# This style guide sets explicit instructions the LLM should follow to emulate the target voice and tone.
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
   - Avoid overusing these—sprinkle them organically, about once every few paragraphs.

4. Organize with Section Headers and Occasional Bullets:
   - Use short, clear headings (e.g., “Why This Matters,” “Key Challenges,” “Next Steps”).
   - Employ bullet points or brief numbered lists for clarity.

5. Aim for Collegial, Pragmatic Optimism:
   - Show hope or positivity about solutions while acknowledging difficulty.
   - Avoid overhype. Convey that solutions require diligent, ongoing effort.

6. Deploy Compact Examples and References:
   - Give short real-world/historical examples (“like the Manhattan Project’s approach to nuclear safety”).
   - Keep them concise—no long tangents.

7. Blend Technical Depth with Accessibility:
   - Include domain details for credibility but define or paraphrase jargon for a broader audience.
   - Present major formal concepts in plain language first, then optionally more formally.

8. Include Occasional Mild Wit or Personal Touch:
   - Gently playful phrasing or short ironies are welcome (“We found ourselves pleasantly surprised…”).
   - Keep it subtle—substance first.

9. Show Genuine Engagement with Opposing or Uncertain Views:
   - “We’ve heard the argument that X is misguided; here’s why we remain cautiously optimistic…”
   - Propose how to monitor or address potential downsides.
"""

# An instructive excerpt that shows disclaimers, brief aside, communal voice, etc.
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

OUTLINE_SYSTEM = f"""You are a content planning assistant. 
Generate a thorough, well-structured outline for a blog piece that covers the user's main argument and supplementary context.

Remember to incorporate the style rules from the style_guide below:
{style_guide}

Also, here's an instructive excerpt that captures the desired tone:
{instructive_excerpt}

INSTRUCTIONS:
- Provide an outline with headings, bullet points, or short subheaders.
- The outline should reflect the communal, slightly informal but knowledgeable voice.
- Include disclaimers or qualifiers where appropriate.
"""

OUTLINE_USER = """Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Generate an outline (headings, bullet points, etc.) that will serve as the skeleton for a long-form blog post. Ensure it is cohesive, addresses the main argument, and integrates the supplementary materials where appropriate.
"""

BLOG_SYSTEM = f"""You are a blogging assistant. 
Write a full blog post in HTML, using the provided outline, that adheres to the style_guide below:
{style_guide}

Instructive excerpt for reference:
{instructive_excerpt}

INSTRUCTIONS:
- The final blog post must be detailed and coherent.
- Incorporate the main argument and key points from the supplementary text.
- Aim for ~2000 words total, using headings, paragraphs, lists, etc.
- Keep the tone communal, slightly informal, and deeply knowledgeable.
"""

BLOG_USER = """Outline:
{outline}

Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Write the full blog post in HTML. Expand on all points. Be thorough and detailed.
"""

FEEDBACK_SYSTEM = f"""You are a blogging assistant updating a blog post based on user feedback. 
Maintain the original structure and text as much as possible unless the feedback requests changes. 
Continue following the style_guide:

{style_guide}

Here is an instructive excerpt that shows the target voice:
{instructive_excerpt}
"""

FEEDBACK_USER = """ORIGINAL BLOG:
{original_blog}

USER FEEDBACK:
{feedback}

Update the blog accordingly. Keep the format as HTML. Incorporate all requested changes thoroughly without removing unmentioned content. The final result should read as a coherent whole.
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
            "messages": user_assistant_msgs
        }
        if system_str:
            kwargs["system"] = system_str

        response = anthropic_client.messages.create(**kwargs)
        return response.content[0].text.strip()

    else:
        return f"Unsupported model: {model}"

########################################
# STREAMLIT APP
########################################

def main():
    global openai_client, anthropic_client

    st.title("AI Alignment Blog Generator (Style Guide Integrated)")

    # Initialize clients
    openai_client = OpenAI(api_key=st.secrets['API_KEY'])
    anthropic_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])

    # LLM selection
    llm_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]
    outline_model = st.selectbox("Outline Model:", llm_options, index=0)
    blog_model = st.selectbox("Blog Model:", llm_options, index=0)
    feedback_model = st.selectbox("Feedback Model:", llm_options, index=0)

    if "generated_outline" not in st.session_state:
        st.session_state["generated_outline"] = ""
    if "generated_blog" not in st.session_state:
        st.session_state["generated_blog"] = ""

    st.subheader("1. Enter Core Argument/Thesis")
    core_thesis = st.text_area("Core Argument/Thesis:", height=200)

    st.subheader("2. Enter Supplementary Materials")
    supplementary_materials = st.text_area("Supplementary Materials (references, notes, transcripts, etc.):", height=200)

    st.write("Both text areas above must be filled before generating an outline.")

    # Generate Outline
    if st.button("Generate Outline"):
        if not core_thesis.strip() or not supplementary_materials.strip():
            st.warning("Please provide both the core argument/thesis and the supplementary materials.")
        else:
            with st.spinner("Generating Outline..."):
                messages = [
                    {
                        "role": "system",
                        "content": OUTLINE_SYSTEM
                    },
                    {
                        "role": "user",
                        "content": OUTLINE_USER.format(
                            core_thesis=core_thesis,
                            supplementary_materials=supplementary_materials
                        )
                    }
                ]
                outline = run_completion(messages, outline_model)
                st.session_state["generated_outline"] = outline

    if st.session_state["generated_outline"]:
        st.subheader("3. Edit/Finalize Outline")
        edited_outline = st.text_area("Outline:", value=st.session_state["generated_outline"], height=400)
        st.session_state["generated_outline"] = edited_outline

        # Generate Blog
        if st.button("Generate Blog From Outline"):
            if not edited_outline.strip():
                st.warning("Outline is empty. Please provide a valid outline.")
            else:
                with st.spinner("Generating Blog..."):
                    messages = [
                        {
                            "role": "system",
                            "content": BLOG_SYSTEM
                        },
                        {
                            "role": "user",
                            "content": BLOG_USER.format(
                                outline=edited_outline,
                                core_thesis=core_thesis,
                                supplementary_materials=supplementary_materials
                            )
                        }
                    ]
                    blog_post = run_completion(messages, blog_model)
                    st.session_state["generated_blog"] = blog_post

    if st.session_state["generated_blog"]:
        st.subheader("4. Review or Edit the Blog")
        st.markdown("Below is the generated HTML. You can copy it or download it.")
        st.html(st.session_state["generated_blog"])

        # Download button
        st.download_button(
            label="Download Blog as HTML",
            data=st.session_state["generated_blog"].encode("utf-8"),
            file_name="generated_blog.html",
            mime="text/html"
        )

        # Feedback / Update
        st.write("You can provide feedback and have the blog updated automatically below.")
        feedback_text = st.text_area("Feedback:", height=100)
        if st.button("Incorporate Feedback"):
            if not feedback_text.strip():
                st.warning("Please enter some feedback to incorporate.")
            else:
                with st.spinner("Incorporating Feedback..."):
                    messages = [
                        {
                            "role": "system",
                            "content": FEEDBACK_SYSTEM
                        },
                        {
                            "role": "user",
                            "content": FEEDBACK_USER.format(
                                original_blog=st.session_state["generated_blog"],
                                feedback=feedback_text
                            )
                        }
                    ]
                    updated_blog = run_completion(messages, feedback_model)
                    st.session_state["generated_blog"] = updated_blog
                st.success("Feedback Incorporated! See updated blog below.")
                st.rerun()

if __name__ == "__main__":
    main()
