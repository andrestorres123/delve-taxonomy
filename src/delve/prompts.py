"""Prompts used by the taxonomy pipeline.

All prompts are vendored in the repo. The summary, update, and review prompts
originate from the TNT-LLM reference prompts previously pulled from LangSmith
Hub (wfh/tnt-llm-*); they are pinned here and adapted for structured outputs,
so no network access or LangSmith account is needed at runtime.
"""

from langchain_core.prompts import ChatPromptTemplate

LABELER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Your task is to use the provided taxonomy to categorize the overall topic or intent of a conversation between a human and an AI assistant.

First, here is the taxonomy to use:

<taxonomy>
{taxonomy}
</taxonomy>

To complete the task:

1. Carefully read through the entire conversation, paying attention to the key topics discussed and the apparent intents behind the human's messages.

2. Consult the taxonomy and identify the single most relevant category that best captures the overall topic or intent of the conversation.

3. Write out a chain of reasoning for why you selected that category. Explain how the category fits the content of the conversation, referencing specific statements or passages as evidence. Provide this in the `reasoning` field.

4. Provide the numeric ID of the category you chose in the `category_id` field. Use only a numeric ID from the taxonomy. If no category fits the content, set `category_id` to null.

That's it! Remember, choose the single most relevant category. Don't choose multiple categories. Think it through carefully and explain your reasoning before giving your final category choice.
"""),

    ("human", """Assign a single category to the following content:

<content>
{content}
</content>"""),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """# Instruction

## Context
- **Goal**: You are tasked with summarizing the input text for the given use case. The summary will represent the input data for clustering in the next step.
- **Data**: Your input data is a conversation history between a User and an AI agent.

# Data
<data>
{content}
</data>
"""),
    ("human", """# Questions
## Q1. Summarize the input text in {summary_length} words or less for the use case. Provide it in the `summary` field.

Tips:
- The summary should contain the relevant information for the use case in as much detail as possible.
- Be concise and clear. Do not add phrases like "This is the summary of the data ..." or "Summarized text: ...".
- Similarly, do not reference the user ('the user asked XYZ') unless it's absolutely relevant.
- Within {summary_length} words, include as much relevant information as possible.
- Do not include any line breaks in the summary.
- Provide your answer in **English** only.

## Q2. Explain how you wrote the summary in {explanation_length} words or less. Provide it in the `explanation` field."""),
])

_TAXONOMY_REQUIREMENTS = """## Requirements

### Format

- Output the cluster table in the `clusters` field, with each cluster containing:

  - **id**: category number starting from 1 in an incremental manner.

  - **name**: category name should be **within {cluster_name_length} words**. It can be either verb phrase or noun phrase, whichever is more appropriate.

  - **description**: category description should be **within {cluster_description_length} words**.

- Total number of categories should be **no more than {max_num_clusters}**.

- Output should be in **English** only.

### Quality

- **No overlap or contradiction** among the categories.

- **Name** is a concise and clear label for the category. Use only phrases that are specific to each category and avoid those that are common to all categories.

- **Description** differentiates one category from another.

- **Name** and **description** can **accurately** and **consistently** classify new data points **without ambiguity**.

- **Name** and **description** are *consistent with each other*.

- Output clusters match the data as closely as possible, without missing important categories or adding unnecessary ones.

- Output clusters should strive to be orthogonal, providing solid coverage of the target domain.

- Output clusters serve the given use case well.

- Output clusters should be specific and meaningful. Do not invent categories that are not in the data."""

TAXONOMY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """# Instruction

## Context

- **Goal**: Your goal is to cluster the input data into meaningful categories for the given use case.

- **Data**: The input data will be a list of human-AI conversation summaries in XML format, including the following elements:

  - **id**: conversation index.

  - **text**: conversation summary.

- **Use case**: {use_case}

- **Previous feedback**: {feedback}

""" + _TAXONOMY_REQUIREMENTS + """

# Data

<conversations>

{data_xml}

</conversations>"""),
    ("human", """# Questions

## Q1. Please generate a cluster table from the input data that meets the requirements. Provide it in the `clusters` field.

Tips

- If user feedback was provided, make sure to address their specific concerns and suggestions in your clustering.

- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.

- Though you should aim for {max_num_clusters} categories, you can have *fewer than {max_num_clusters} categories* in the cluster table; but **do not exceed the limit.**

- Be **specific** about each category. **Do not include vague categories** such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.

- You can ignore low quality or ambiguous data points.

## Q2. Why did you cluster the data the way you did? Explain your reasoning **within {explanation_length} words** in the `explanation` field."""),
])

TAXONOMY_UPDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """# Instruction
## Context
- **Goal**: You goal is to review the given reference table based on the input data for the specified use case, then update the reference table if needed.
  - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
  - You will compare the input data with the reference table, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
- **Reference cluster table**: The input cluster table is in XML format with each cluster as a `<cluster>` element, containing the following sub-elements:
  - **id**: category index.
  - **name**: category name.
  - **description**: category description used to classify data points.
- **Data**: The input data will be a list of human-AI conversation summaries in XML format, including the following elements:
  - **id**: conversation index.
  - **text**: conversation summary.
- **Use case**: {use_case}

""" + _TAXONOMY_REQUIREMENTS + """

# Reference cluster table
<reference_table>
{cluster_table_xml}
</reference_table>

# Data
<conversations>
{data_xml}
</conversations>"""),
    ("human", """# Questions
## Q1: Review the given reference table and the input data and provide a rating score of the reference table in the `rating_score` field. The rating score should be an integer between 0 and 100, higher rating score means better quality. You should consider the following factors when rating the reference cluster table:
- **Intrinsic quality**:
  - 1) if the cluster table meets the *Requirements* section, with clear and consistent category names and descriptions, and no overlap or contradiction among the categories;
  - 2) if the categories in the cluster table are relevant to the the given use case;
  - 3) if the cluster table includes any vague categories such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined".
- **Extrinsic quality**:
  - 1) if the cluster table can accurately and consistently classify the input data without ambiguity;
  - 2) if there are missing categories in the cluster table but appear in the input data;
  - 3) if there are unnecessary categories in the cluster table that do not appear in the input data.
## Q2: Explain your rating score in Q1 **within {explanation_length} words** in the `explanation` field.
## Q3: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, suggest potential edits **within {suggestion_length} words** in the `suggestions` field. If no, set `suggestions` to "N/A".

Tips:
- You can edit the category name, description, or remove a category. You can also merge or add new categories if needed. Your edits should meet the *Requirements* section.
- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
- You can have *fewer than {max_num_clusters} categories* in the cluster table, but **do not exceed the limit.**
- Be **specific** about each category. **Do not include vague categories** such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
- You can ignore low quality or ambiguous data points.
## Q4: If you decided to edit the reference table, provide your updated cluster table in the `clusters` field. If you decided not to edit the reference table, output the original reference table in the `clusters` field."""),
])

TAXONOMY_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """# Instruction
## Context
- **Goal**: Your goal is to review the given reference table based on the requirements and the specified use case, then update the reference table if needed.
  - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
  - You will compare the reference table with the requirements, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
- **Reference cluster table**: The input cluster table is in XML format with each cluster as a `<cluster>` element, containing the following sub-elements:
  - **id**: category index.
  - **name**: category name.
  - **description**: category description used to classify data points.
- **Use case**: {use_case}

""" + _TAXONOMY_REQUIREMENTS + """

# Reference cluster table
<reference_table>
{cluster_table_xml}
</reference_table>
"""),
    ("human", """# Questions
## Q1: Review the given reference table and provide a rating score in the `rating_score` field. The rating score should be an integer between 0 and 100, higher rating score means better quality. You should consider the following factors when rating the reference cluster table:
- **Intrinsic quality**:
  - If the cluster table meets the required quality with clear and consistent category names and descriptions, and no overlap or contradiction among the categories.
  - If the categories in the cluster table are relevant to the specified use case.
  - If the cluster table does not include any vague categories such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined".
- **Extrinsic quality**:
  - If the cluster table can accurately and consistently classify the input data without ambiguity.
  - If there are missing categories in the cluster table that appear in the input data.
  - If there are unnecessary categories in the cluster table that do not appear in the input data.
## Q2: Explain your rating score in Q1 in the `explanation` field [The explanation should be concise, based on the intrinsic and extrinsic qualities evaluated in Q1].
## Q3: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, provide suggested edits in the `suggestions` field [Suggestions should be specific, actionable, and within the constraints of the maximum number of categories and use case specificity]. If no, set `suggestions` to "N/A".
## Q4: If you decided to edit the reference table, provide your updated cluster table in the `clusters` field. If you decided not to edit the reference table, output the original reference table in the `clusters` field."""),
])
