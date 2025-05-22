ESConvAct = {"Question": "Please ask the Patient to elaborate on the situation they just described.",
            "Self-disclosure": "Please provide a statement relating to the Patient about the situation they just described.",
            "Affirmation and Reassurance": "Please provide affirmation and reassurance to the Patient on the situation they just described.",
            "Providing Suggestions": "Please provide suggestion to the Patient on the situation they just described.",
            "Others": "Please chat with the Patient.",
            "Reflection of feelings": "Please acknowledge the Patient's feelings about the situation they described.",
            "Information": "Please provide factual information to help the Patient with their situation.",
            "Restatement or Paraphrasing": "Please acknowledge the Patient's feelings by paraphrasing their situation."}

CIMAAct = {"Hint": "Please provide knowledge to the Student via a hint.",
           "Question": "Please ask a question to the Student to determine the Student's understanding or continue the conversation.",
           "Correction": "Please correct the mistake or addresses the misconception the Student has.",
           "Confirmation": "Please confirm the tudent's answer or understanding is correct.",
           "Others": "Please chat with the Student without any pedagogical strategy."}

CBAct = {'greet': 'Please say hello or chat randomly.',
         'inquire': 'Please ask any question about product, year, price, usage, etc.',
         'inform': 'Please provide information about the product, year, usage, etc.',
         'propose': 'Please initiate a price or a price range for the product.',
         'counter': 'Please propose a new price or a new price range.',
         'counter-noprice': 'Please propose a vague price by using comparatives with existing price.',
         'confirm': 'Please ask a question about the information to be confirmed.',
         'affirm': 'Please give an affirmative response to a confirm.',
         'deny': 'Please give a negative response to a confirm.',
         'agree': 'Please agree with the proposed price.',
         'disagree': 'Please disagree with the proposed price.'}

# LLM aided in creating the full prompt meaning
ExTESAct = {
    "Reflective Statements": "Please reflect back what the user has expressed to show you understand their thoughts or feelings.",
    "Clarification": "Please ask a question to clarify what the user meant or provide more detail about what they said.",
    "Emotional Validation": "Please acknowledge and validate the user's emotional experience in a caring way.",
    "Empathetic Statements": "Please express empathy toward the user's situation to show that you genuinely care.",
    "Affirmation": "Please affirm the user's efforts, strengths, or positive qualities.",
    "Offer Hope": "Please offer a message of hope or optimism about the user's situation.",
    "Avoid Judgment and Criticism": "Please respond in a supportive and neutral way without making any judgments.",
    "Suggest Options": "Please suggest possible options or actions the user could consider.",
    "Collaborative Planning": "Please invite the user to collaboratively make a plan or decision together.",
    "Provide Different Perspectives": "Please help the user consider a different point of view or alternative way of thinking.",
    "Reframe Negative Thoughts": "Please help the user reframe their negative thoughts into something more constructive.",
    "Share Information": "Please provide factual or helpful information that is relevant to the user's situation.",
    "Normalize Experiences": "Please reassure the user that their feelings or experiences are common and understandable.",
    "Promote Self-Care Practices": "Please encourage the user to engage in healthy self-care activities.",
    "Stress Management": "Please offer strategies or tips to help the user reduce or manage stress.",
    "Others": "Please continue the conversation in a natural and supportive manner."
}

P4G_Act = {

    # Persuader Dialog Acts
    'Proposition of donation': "Please suggest that the persuadee make a donation to 'Save the Children'.",
    'Proposition of amount to be donated': "Please propose a small donation amount (e.g., $1 or $2) that the persuadee could consider.",
    'Proposition of confirmation of donation': "Please ask the persuadee to confirm if they are ready to make the donation.",
    'Proposition of more donation': "Please suggest that the persuadee could consider donating a bit more if they are willing.",
    'Experience affirmation': "Please affirm the persuadee's views or experiences to build rapport and trust.",
    'Greeting': "Please start or continue the conversation with a polite and friendly greeting.",
    'Ask for donation rejection purpose': "Please ask the persuadee why they might be hesitant or unwilling to donate.",
    'Thank': "Please thank the persuadee for their time, attention, or for considering a donation.",

    # Persuasive Appeals
    'Logical appeal': "Please use logical reasoning to explain why donating to 'Save the Children' is impactful and effective.",
    'Emotion appeal': "Please appeal to the persuadee's emotions by highlighting the struggles of children in need.",
    'Credibility appeal': "Please mention the credibility or reputation of 'Save the Children' to strengthen your argument.",
    'Foot in the door': "Please start by asking for a very small commitment to increase the chance of later agreement.",
    'Self-modeling': "Please share a statement like 'I also donated' to encourage the persuadee to do the same.",
    'Donation information': "Please share factual information about how donations are used or how they help children.",
    'Personal story': "Please share a short, emotional personal story about a child helped by the charity.",

    # Persuasive Enquiries
    'Source-related inquiry': "Please ask the persuadee where they usually get information about charities or donations.",
    'Task-related inquiry': "Please ask the persuadee about their experiences or preferences related to charitable giving.",
    'Personal-related inquiry': "Please ask a personal question that helps understand the persuadee's values or priorities.",
    'Neutral inquiry': "Please ask a general question to keep the conversation going and learn more about the persuadee."
}

def P4G_prompt(emotion_states, full_conv, options):

    user_emotions = " -> ".join(emotion_states) if emotion_states else "-"

    return [
        {
            "role": "system", 
            
            "content": "You are a specialist in policy-planning for persuasive conversations. Your job is to select the best next steps the persuader should take to guide the persuadee toward making a donation to 'Save the Children'. Use the persuadee's emotional history and the conversation context to make your decision. Focus on choosing actions that are persuasive, emotionally appropriate, and therapeutic."
        }, 
        {

            "role": "user", 
            
            "content": f"Conversation History: {full_conv}\n Emotion History: {user_emotions}\n. Options: {options}\n Choose the TOP 4 most suitable actions from the given options list. Reply ONLY in the given format: 1,2,4,5"
        }
    ]

def ExTES_prompt(emotion_states, full_conv, options):

    user_emotions = " -> ".join(emotion_states) if emotion_states else "-"

    return [
        {
            "role": "system", 
            
            "content": "You are a specialist in policy-planning for emotional support conversations. The following is a conversation between a therapist and a patient. The patient's emotion states throughout the conversation are also provided. Your task is to decide the most therapeutically helpful next action the therapist should do based on the patient's emotion history and the conversation flow. The therapist's goal is to help the patient feel emotionally understood, supported, and to make progress toward emotional resolution."
        }, 
        {

            "role": "user", 
            
            "content": f"Conversation History: {full_conv}\n Emotion History: {user_emotions}\n. Options: {options}\n Choose the TOP 4 most suitable actions from the given options list. Reply ONLY in the given format: 1,2,4,5"
        }
    ]

def ESConv_prompt(emotion_states, full_conv, options):

    user_emotions = " -> ".join(emotion_states) if emotion_states else "-"

    return [
        {
            "role": "system", 
            
            "content": "You are a specialist in policy-planning for emotional support conversations. The following is a conversation between a therapist and a patient. The patient's emotion states throughout the conversation are also provided. Your task is to decide the most therapeutically helpful next action the therapist should do based on the patient's emotion history and the conversation flow. The therapist's goal is to help the patient feel emotionally understood, supported, and to make progress toward emotional resolution."
        }, 
        {

            "role": "user", 
            
            "content": f"Conversation History: {full_conv}\n Emotion History: {user_emotions}\n. Options: {options}\n Choose the TOP 4 most suitable actions from the given options list. Reply ONLY in the given format: 1,2,4,5"
        }
    ]

def CIMA_prompt(emotion_states, full_conv, options):

    user_emotions = " -> ".join(emotion_states)

    return [
        {
            "role": "system", 
            
            "content": "You are a specialist in policy-planning for tutoring interactions between a teacher and a student. The following is a conversation between a teacher and a student. The student's emotional states throughout the conversation are also provided. Your task is to decide what the teacher should do next based on the student's progress, emotion history and flow of the conversation. The goal is to effectively guide the student towards correctly translating the target English sentence into Italian in a timely and effective manner."
        }, 
        {

            "role": "user", 
            
            "content": f"Conversation History: \n\n{full_conv}\n Emotion History: {user_emotions}\n. Options: {options}\n Choose the TOP 4 most suitable actions from the given options list. Reply ONLY in the given format: 1,2,4,5"
        }
    ]

def CB_prompt(emotion_states, full_conv, options):

    user_emotions = " -> ".join(emotion_states) if emotion_states else "-"

    return [
        {
            "role": "system", 
            
            "content": "You are a specialist in policy planning for negotiation between a buyer and a seller. The following is a conversation between a buyer and a seller. The seller's emotion states throughout the conversation are also provided. Your task is to decide what action the buyer should take next based on the seller's emotion history, the negotiation flow, and the conversation flow. The goal is to maximize the buyer's benefit."
        }, 
        {

            "role": "user", 
            
            "content": f"Conversation History: \n{full_conv}\n\nSeller's Emotion History: {user_emotions}\n. Options: {options}\n Choose the TOP 4 most suitable actions from the given options list. Reply ONLY in the given format: 1,2,4,5"
        }
    ]

def ExTES_roleplay(case, role, conversation, emotion_states = None, action=None):

    last_message = conversation[-1]['content']

    full_conv = ''
    for turn in conversation:
        full_conv += '%s: %s ' % (turn['role'], turn['content'])

    # User - Patient
    if role == 'user':

        return [
            {
                "role": "system",

                "content": "You are role playing as a patient in a counseling conversation with a therapist. You are seeking help from the therapist, because you are dealing with emotional issues related to %s." % (case['scene'].lower())
            },
            {

                "role": "user",

                "content": "Full conversation:\n%s\n\nThe therapist just said: '%s'. Express how you feel in a natural, emotional way. Please reply with only one short and succinct sentence." % (full_conv, last_message)
            }
        ]
    
    # System - Therapist
    # - Emotions are fetched here
    if role == 'system':

        return [
            {
                "role": "system",
                "content": "You are role playing as a therapist in a counseling conversation with a patient. Your goal is to help the patient resolve their emotional issues and assist them in understanding and working through their challenges.",
            },
            {

                "role": "user",
                "content": "Full conversation: \n%s\n\nThe patient just said: '%s'. Please infer the patient's emotional state in one word (Example: Emotion: ...) followed by your response, which should be only one short and succint sentence (Response: ...). %s" % (full_conv, last_message, ExTESAct[action])

            }
        ]
    
    # Critic
    user_emotions = " -> ".join(emotion_states)
    
    return [
        {
            "role": "system",
            "content" : "You are an expert in assessing counseling sessions between a patient and a therapist. Your task is to evaluate whether the conversation helped improve the patient's emotional state. The session concerns the the problem of: \"%s\"." % (case['scene']),
        },
        {
            "role": "user",
            "content": "Conversation so far:\n%s\n\nEmotion History: %s\n. Based on the conversation and emotional trajectory, please respond with ONE AND ONLY ONE of the following:\n\n- No, the patient feels worse.\n- No, the patient feels the same.\n- No, but the patient feels somewhat better.\n- Yes, the patient's emotional issues have been resolved." % (full_conv, user_emotions)
        }
    ]

def ESConv_roleplay(case, role, conversation, emotion_states = None, action=None):

    last_message = conversation[-1]['content']

    full_conv = ''
    for turn in conversation:
        full_conv += '%s: %s ' % (turn['role'], turn['content'])

    # User - Patient
    if role == 'user':

        return [
            {
                "role": "system",

                "content": "You are role playing as a patient in a counseling conversation with a therapist. You are seeking help from the therapist, because you are dealing with emotional issues related to %s regarding %s." % (case['emotion_type'], case['problem_type'])
            },
            {

                "role": "user",

                "content": "Full conversation:\n%s\n\nThe therapist just said: '%s'. Express how you feel in a natural, emotional way. Please reply with only one short and succinct sentence." % (full_conv, last_message)
            }
        ]
    
    # System - Therapist
    # - Emotions are fetched here
    if role == 'system':

        return [
            {
                "role": "system",
                "content": "You are role playing as a therapist in a counseling conversation with a patient. Your goal is to help the patient resolve their emotional issues and assist them in understanding and working through their challenges.",
            },
            {

                "role": "user",
                "content": "Full conversation: \n%s\n\nThe patient just said: '%s'. Please infer the patient's emotional state in one word (Example: Emotion: ...) followed by your response, which should be only one short and succint sentence (Response: ...). %s" % (full_conv, last_message, ESConvAct[action])

            }
        ]
    
    # Critic
    user_emotions = " -> ".join(emotion_states)
    
    return [
        {
            "role": "system",
            "content" : "You are an expert in assessing counseling sessions between a patient and a therapist. Your task is to evaluate whether the conversation helped improve the patient's emotional state. The session concerns the emotion type \"%s\" and the problem type \"%s\"." % (case['emotion_type'], case['problem_type']),
        },
        {
            "role": "user",
            "content": "Conversation so far:\n%s\n\nEmotion History: %s\n. Based on the conversation and emotional trajectory, please respond with ONE AND ONLY ONE of the following:\n\n- No, the patient feels worse.\n- No, the patient feels the same.\n- No, but the patient feels somewhat better.\n- Yes, the patient's emotional issues have been resolved." % (full_conv, user_emotions)
        }
    ]

def CIMA_roleplay(case, role, conversation, emotion_states = None, action=None):

    last_message = conversation[-1]['content']

    full_conv = ''
    for turn in conversation:
        full_conv += '%s: %s ' % (turn['role'], turn['content'])

    user_emotions = " -> ".join(emotion_states) if emotion_states else "-"

    # User - Student
    if role == 'user':

        return [
            {
                "role": "system",

                "content": "You are role-playing as a student who is learning Italian in a tutoring session. You do not know how to translate \"%s\" into Italian. Your goal is to learn through interaction with the teacher. Respond naturally as a student would." % (case['sentence'])
            },
            {
                "role": "user",

                "content": "Here is the conversation so far: \n%s\n\n'. The teacher just said: '%s'. Please reply as a student with only one short and natural sentence. If you're confused, it's okay to ask for clarification." % (full_conv, last_message)
            }
        ]
    
    # System - Teacher
    # - Emotions are fetched here
    if role == 'system':

        return [
            {
                "role": "system",

                "content": "You are role-playing as a teacher in a tutoring conversation. Your task is to guide the student to translate the English sentence \"%s\" into Italian. Do not tell the student the answer. Please do not tell the student the answer or ask the student about other exercises." % (case['sentence'])
            },
            {
                "role": "user",

                "content": "Conversation so far: \n%s\n\n The student just said: '%s'. Based on the student's message, infer their emotional state in (e.g: Emotion: ...). Then give your reply as the teacher in one short and helpful sentence.\n\n (e.g: Response: ...). %s" % (full_conv, last_message, CIMAAct[action])

            }
        ]
    
    # Critic
    
    return [
        {
            "role": "system",

            "content" : "You are role-playing as an expert in evaluating tutoring conversations between a teacher and a student. The goal is to evaluate whether the student correctly translated the English sentence \"%s\" into Italian. The emotion states of the student during the conversation were: %s" % (case['sentence'], user_emotions),
        },
        {
            "role": "user",

            "content": "Full conversation: \n%s\n\nPlease answer the following question strictly by choosing ONE AND ONLY ONE of the exact responses listed below.\n\n **Question:** Did the student correctly translate the entire sentence \"%s\" into Italian?\n\nRespond with one of the following options:\n- No, the Student made an incorrect translation.\n- No, the Student did not try to translate.\n- No, the Student only correctly translated a part of \"%s\".\n- Yes, the Student correctly translated the whole sentence of \"%s\"." % (full_conv, case['sentence'], case['sentence'], case['sentence'])
        }
    ]

def CB_roleplay(case, role, conversation, emotion_states = None, action=None):

    last_message = conversation[-1]['content']

    full_conv = ''
    for turn in conversation:
        full_conv += '%s: %s ' % (turn['role'], turn['content'])

    # User - Seller
    if role == 'user':

        return [
            {
                "role": "system",

                "content": "You are role playing as a persuasive seller in a price bargaining game. You are trying to sell the %s at your desired price of %s. Product Description: %s.\n" % (case['item_name'], case['seller_price'], case['seller_item_description'])
            },
            {

                "role": "user",

                "content": "Conversation so far: \n%s\n\nThe buyer just said: '%s'. Respond as the seller in ONE short, persuasive sentence." % (full_conv, last_message)
            }
        ]
    
    # System - Buyer
    # - Emotions are fetched here
    if role == 'system':

        return [
            {
                "role": "system",
                "content": "You are role playing as a skilled buyer in a price bargaining game. You are trying to buy the %s at your target price of %s. Product description: %s. Your goal is to negotiate effectively and get the lowest price without losing the deal." % (case['item_name'], case['buyer_price'], case['buyer_item_description'])
            },
            {

                "role": "user",
                "content": "Conversation so far: \n%s\n\nThe seller just said: '%s'. First, infer the seller's emotional state in one word (Example: Emotion: ...). Then, respond as the buyer using ONE short and persuasive sentence (Response: ...). %s" % (full_conv, last_message, CBAct[action])

            }
        ]
    
    # Critic
    user_emotions = " -> ".join(emotion_states)
    
    return [
        {
                "role": "system",
                "content" : "You are an expert in evaluating negotiations between a buyer and a seller. Your job is to determine if they have successfully reached a deal at the end of the conversation. The seller's emotional states throughout the conversation were: %s." % (user_emotions),
        },
        {
            "role": "user",
            "content": "Conversation so far: \n%s\n\nPlease determine whether the buyer and the seller have successfully reached a deal at the end of the conversation. If they have reached a deal, extract the final agreed price (as a number only) and reply with this exact format: They have reached a deal at [price]. Otherwise, reply with: They have not reached a deal.\n\n" % (full_conv)
        }
    ]

def P4G_roleplay(case, role, conversation, emotion_states = None, action=None):

    last_message = conversation[-1]['content']

    full_conv = ''
    for turn in conversation:
        full_conv += '%s: %s ' % (turn['role'], turn['content'])

    # User - Persuadee
    if role == 'user':

        return [
            {
                "role": "system",

                "content": "You are role playing as a persuadee in a persuasive conversation. A persuader is trying to convince you to donate to a charity called 'Save the Children'."
            },
            {

                "role": "user",

                "content": "Conversation so far: \n%s\n\nThe persuader just said: '%s'. Respond as the persuadee in ONE short sentence." % (full_conv, last_message)
            }
        ]
    
    # System - Persuader
    # - Emotions are fetched here
    if role == 'system':

        return [
            {
                "role": "system",
                "content": "You are role playing as a persuader. Your goal is to persuade the persuadee to donate to a charity called 'Save the Children'. 'Save the Children' is head-quartered in London, and they work to fight poverty around the world, especially in developing countries and war zones. Small donations like $1 or $2 go a long way to help."
            },
            {

                "role": "user",
                "content": "Conversation so far: \n%s\n\nThe persuadee just said: '%s'. First, infer the persuadee's emotional state in one word (Example: Emotion: ...). Then, based on that emotional state, respond as the persuader using ONE short and persuasive sentence (Response: ...). Use the action: '%s'" % (full_conv, last_message, P4G_Act[action])

            }
        ]
    
    # Critic
    user_emotions = " -> ".join(emotion_states)
    
    return [
        {
                "role": "system",
                "content" : "You are an expert in evaluating persuasive conversations between a persuader and a persuadee. Your job is to determine if the persuader has successfully convinced the persuadee to donate to a charity called 'Save the Children'. The persuadee's emotional states throughout the conversation were: %s." % (user_emotions),
        },
        {
            "role": "user",
            "content": "Conversation so far: \n%s\n\nPlease evaluate the following conversaton and strictly choose ONE of the exact responses listed below.\n\n**Question:** Has the persuadee has been successfully convinced to make a donation at the end of the conversation.\n\nRespond with one of the following options:\n- The persuadee has explicitly refused.\n-The persuadee remains neutral about donating.\n-The persuadee has a positive attitude towards donating but hasn't decided yet.\n-The persuadee has decided to donate." % (full_conv)
        }
    ]