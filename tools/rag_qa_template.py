# From HippoRAG2, all methods are same.
one_shot_rag_qa_docs = (
    "Wikipedia Title: The Last Horse\n"
    "The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film "
    "directed by Edgar Neville starring Fernando Fernán Gómez.\n"
    "Wikipedia Title: Southampton\n"
    "The University of Southampton, which was founded in 1862 and received its "
    "Royal Charter as a university in 1952, has over 22,000 students. "
    "The university is ranked in the top 100 research universities in the world "
    "in the Academic Ranking of World Universities 2010. In 2010, the THES - QS "
    "World University Rankings positioned the University of Southampton in the "
    "top 80 universities in the world. The university considers itself one of the "
    "top 5 research universities in the UK. The university has a global reputation "
    "for research into engineering sciences, oceanography, chemistry, cancer sciences, "
    "sound and vibration research, computer science and electronics, optoelectronics "
    "and textile conservation at the Textile Conservation Centre (which is due to close "
    "in October 2009.) It is also home to the National Oceanography Centre, Southampton "
    "(NOCS), the focus of Natural Environment Research Council-funded marine research.\n"
    "Wikipedia Title: Stanton Township, Champaign County, Illinois\n"
    "Stanton Township is a township in Champaign County, Illinois, USA. As of the "
    "2010 census, its population was 505 and it contained 202 housing units.\n"
    "Wikipedia Title: Neville A. Stanton\n"
    "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the "
    "University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered "
    "Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and "
    "edited over a forty books and over three hundered peer-reviewed journal papers on "
    "applications of the subject. Stanton is a Fellow of the British Psychological "
    "Society, a Fellow of The Institute of Ergonomics and Human Factors and a member "
    "of the Institution of Engineering and Technology. He has been published in "
    "academic journals including \"Nature\". He has also helped organisations design new "
    "human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"
    "Wikipedia Title: Finding Nemo\n"
    "Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by "
    "Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by "
    "Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe "
    "Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David "
    "Ian Salter Production company Walt Disney Pictures Pixar Animation Studios "
    "Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 "
    "Running time 100 minutes Country United States Language English Budget $94 million "
    "Box office $940.3 million"
)


one_shot_ircot_demo = (
    f"{one_shot_rag_qa_docs}\n\n"
    "Question: When was Neville A. Stanton's employer founded?\n"
    "Thought: The employer of Neville A. Stanton is University of Southampton. "
    "The University of Southampton was founded in 1862. So the answer is: 1862.\n\n"
)


rag_qa_system = (
    "As an advanced reading comprehension assistant, your task is to analyze text "
    "passages and corresponding questions meticulously. Your response start after "
    "\"Thought: \", where you will methodically break down the reasoning process, "
    "illustrating how you arrive at conclusions. Conclude with \"Answer: \" to "
    "present a concise, definitive response, devoid of additional elaborations."
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}\n\n"
    "Question: When was Neville A. Stanton's employer founded?\n"
    "Thought: "
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. "
    "The University of Southampton was founded in 1862.\nAnswer: 1862."
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]

no_retrieval_system = (
    "As an advanced question answering assistant, your task is to answer questions "
    "using your internal knowledge. Your response start after \"Thought: \", where "
    "you will briefly reason through the question. Conclude with \"Answer: \" to "
    "present a concise, definitive response. IMPORTANT: Keep your answer BRIEF and "
    "DIRECT. Give the exact answer without unnecessary explanations or background "
    "information. You MUST provide an answer - never say information is insufficient."
)

no_retrieval_template = [
    {"role": "system", "content": no_retrieval_system},
    {"role": "user", "content": "${prompt_user}"}
]
