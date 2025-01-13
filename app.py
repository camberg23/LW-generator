# alignment_blog_app.py

import streamlit as st
import re
import anthropic
from openai import OpenAI

########################################
# MODEL UTILS
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

def run_completion(messages, model, openai_client, anthropic_client, verbose=False):
    if is_o1_model(model) or is_gpt4o_model(model):
        # For O1 and GPT-4o models via openai
        if is_o1_model(model):
            messages = adjust_messages_for_o1(messages)
        if verbose:
            st.write(f"**Using OpenAI Model:** {model}")
            st.write("**Messages:**", messages)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        response = completion.choices[0].message.content
        if verbose:
            st.write("**Response:**", response)
        return response
    elif is_anthropic_model(model):
        # For Anthropic
        if verbose:
            st.write(f"**Using Anthropic Model:** {model}")
            st.write("**Messages:**", messages)
        system_str_list = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] != "system"]
        system_str = "\n".join(system_str_list) if system_str_list else None
        # Example usage of anthropic for a Claude model
        # Adjust model parameter to match your environment
        kwargs = {
            "model": "claude-instant-1",
            "max_tokens_to_sample": 1000,
            "messages": []
        }
        # Anthropic expects user->assistant->user->assistant...
        # We'll place the system text at the top, then alternate user/assistant from messages
        full_convo = ""
        if system_str:
            full_convo += f"{anthropic.HUMAN_PROMPT}System: {system_str}{anthropic.AI_PROMPT}"
        for m in user_assistant_msgs:
            if m["role"] == "user":
                full_convo += f"{anthropic.HUMAN_PROMPT}{m['content']}"
            else:
                full_convo += f"{anthropic.AI_PROMPT}{m['content']}"
        full_convo += anthropic.AI_PROMPT
        kwargs["prompt"] = full_convo
        response = anthropic_client.completions.create(**kwargs)
        if verbose:
            st.write("**Response:**", response.completion)
        return response.completion.strip()
    else:
        return "Unsupported model: " + model

########################################
# PROMPTS
########################################

OUTLINE_SYSTEM = """You are a content planning assistant. Generate a comprehensive, well-structured outline for a blog piece that covers the user's main argument and supplementary context. The outline should be sufficiently detailed so a writer can follow it to write a full blog."""
OUTLINE_USER = """Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Generate an outline (headings, bullet points, etc.) that will serve as the skeleton for a long-form blog post. Ensure it is cohesive, addresses the main argument, and integrates the supplementary materials where appropriate.
"""

BLOG_SYSTEM = """You are a blogging assistant. Write a full blog post in HTML format, using the provided outline. The content should be detailed, coherent, and must incorporate the main argument and any key points from the supplementary text. Aim for ~2000 words total. Use headings, paragraphs, lists, and other HTML elements as needed. Do not hold back on detail."""
BLOG_USER = """Outline:
{outline}

Main Argument/Thesis:
{core_thesis}

Supplementary Materials:
{supplementary_materials}

Write the full blog post in HTML. Expand on all points. Be thorough and detailed.
"""

FEEDBACK_SYSTEM = """You are a blogging assistant updating a previously written blog post based on new user feedback. Retain as much of the original structure and text as possible unless changes are requested by the feedback."""
FEEDBACK_USER = """ORIGINAL BLOG:
{original_blog}

USER FEEDBACK:
{feedback}

Update the blog accordingly. Keep the format as HTML. Incorporate all requested changes thoroughly without removing unmentioned content. The final result should read as a coherent whole.
"""

########################################
# STREAMLIT APP
########################################

def main():
    st.title("AI Alignment Blog Generator")

    # LLM selection
    llm_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]
    outline_model = st.selectbox("Outline Model:", llm_options, index=0)
    blog_model = st.selectbox("Blog Model:", llm_options, index=0)
    feedback_model = st.selectbox("Feedback Model:", llm_options, index=0)

    # Initialize clients
    openai_client = OpenAI(api_key=st.secrets['API_KEY'])
    anthropic_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])

    # Session states
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
                    {"role": "system", "content": OUTLINE_SYSTEM},
                    {"role": "user", "content": OUTLINE_USER.format(core_thesis=core_thesis, supplementary_materials=supplementary_materials)}
                ]
                outline = run_completion(messages, outline_model, openai_client, anthropic_client)
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
                        {"role": "system", "content": BLOG_SYSTEM},
                        {"role": "user", "content": BLOG_USER.format(
                            outline=edited_outline,
                            core_thesis=core_thesis,
                            supplementary_materials=supplementary_materials
                        )}
                    ]
                    blog_post = run_completion(messages, blog_model, openai_client, anthropic_client)
                    st.session_state["generated_blog"] = blog_post

    if st.session_state["generated_blog"]:
        st.subheader("4. Review or Edit the Blog")
        st.markdown("Below is the generated HTML. You can copy it or download it.")

        # Display the blog inline as HTML (no iframe)
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
                        {"role": "system", "content": FEEDBACK_SYSTEM},
                        {"role": "user", "content": FEEDBACK_USER.format(
                            original_blog=st.session_state["generated_blog"],
                            feedback=feedback_text
                        )}
                    ]
                    updated_blog = run_completion(messages, feedback_model, openai_client, anthropic_client)
                    st.session_state["generated_blog"] = updated_blog
                st.success("Feedback Incorporated! See updated blog below.")
                st.rerun()

if __name__ == "__main__":
    main()
