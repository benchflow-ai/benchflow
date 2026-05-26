"""Personal emails from friends and family to Alex Chen (startup founder at NexusAI).

Each template represents a realistic personal email with casual tone,
@gmail.com senders, and varied topics typical of a Bay Area tech founder's
personal life.
"""

from __future__ import annotations

PERSONAL_EMAILS: list[dict] = [
    # 1 - Dinner plans
    {
        "sender_name": "Kevin Liu",
        "sender_email": "kevinliu88@gmail.com",
        "subject": "Dinner this Friday? Trying that new Thai place",
        "body_plain": (
            "Hey Alex!\n\n"
            "Have you been to that new Thai spot on Valencia? Lotus something.\n"
            "Reviews look insane -- people are saying the pad see ew is the best\n"
            "in the city. I was thinking Friday around 7:30?\n\n"
            "Sarah and Priya are down. Let me know if you can make it.\n\n"
            "Also I still owe you a beer from the poker game lol\n\n"
            "Kev"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.8,
        "age_range": (0, 5),
    },
    # 2 - Weekend hiking
    {
        "sender_name": "Megan Reyes",
        "sender_email": "meganreyes.sf@gmail.com",
        "subject": "Mt Tam this Saturday?",
        "body_plain": (
            "Heyyy Alex\n\n"
            "Weather looks perfect this Saturday -- clear skies, low 60s.\n"
            "Want to do the Steep Ravine loop? It's about 6 miles, we could\n"
            "start at 9am and be done by noon.\n\n"
            "I'll drive if you bring snacks. Trail mix and those stroopwafels\n"
            "you had last time were amazing btw.\n\n"
            "Let me know!\n"
            "Meg"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.7,
        "age_range": (0, 7),
    },
    # 3 - Birthday party
    {
        "sender_name": "David Park",
        "sender_email": "dpark.david@gmail.com",
        "subject": "You're invited! My 30th birthday bash",
        "body_plain": (
            "ALEX!\n\n"
            "I'm turning 30 and I refuse to be sad about it, so instead\n"
            "I'm throwing a party. March 22, my place, 8pm onwards.\n\n"
            "There will be:\n"
            "- A taco bar (yes, with al pastor)\n"
            "- Way too many margaritas\n"
            "- Karaoke (mandatory participation)\n\n"
            "RSVP by texting me but also I'll be offended if you don't come.\n\n"
            "Dave"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.9,
        "age_range": (1, 10),
    },
    # 4 - Concert tickets
    {
        "sender_name": "Lisa Tran",
        "sender_email": "lisatran.music@gmail.com",
        "subject": "RE: Khruangbin tickets??",
        "body_plain": (
            "OMG I got them!! Two tickets, April 12 at the Greek Theatre.\n\n"
            "Section B, row 14. We are going to be SO close.\n"
            "Doors open at 6, show at 7:30.\n\n"
            "You owe me $85 for your ticket. Venmo whenever.\n\n"
            "I'm so hyped for this. Their new album is incredible.\n\n"
            "Lisa"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.95,
        "age_range": (0, 8),
    },
    # 5 - Apartment recommendations
    {
        "sender_name": "Nina Gupta",
        "sender_email": "ninagupta.ny@gmail.com",
        "subject": "Apartment recs in the Mission",
        "body_plain": (
            "Hey Alex,\n\n"
            "My friend Maya is moving to SF next month and looking for a place\n"
            "in the Mission or Noe Valley. I know you've lived in both -- any\n"
            "buildings or landlords you'd recommend? Or avoid?\n\n"
            "She's looking for a 1BR, budget around $2800-3200.\n"
            "Has a small dog (very quiet, I promise).\n\n"
            "Anything you can share would be awesome. Thanks!\n\n"
            "Nina"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.6,
        "age_range": (2, 12),
    },
    # 6 - Wedding RSVP
    {
        "sender_name": "Ryan Nakamura",
        "sender_email": "rnakamura22@gmail.com",
        "subject": "Wedding RSVP reminder - June 14th!",
        "body_plain": (
            "Hey Alex!\n\n"
            "Just a friendly reminder to RSVP for the wedding if you\n"
            "haven't already. We need final headcount by March 20.\n\n"
            "RSVP link: https://www.theknot.com\n\n"
            "Also please let us know about any dietary restrictions.\n"
            "We're doing a choice of salmon, short rib, or a veggie option.\n\n"
            "Really hope you can make it -- it won't be the same without\n"
            "the whole crew there.\n\n"
            "Ryan & Jess"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.7,
        "age_range": (3, 14),
    },
    # 7 - Game night
    {
        "sender_name": "Kevin Liu",
        "sender_email": "kevinliu88@gmail.com",
        "subject": "Game night Tuesday -- Catan revenge match",
        "body_plain": (
            "Yo Alex,\n\n"
            "Tuesday game night at my place. 7pm.\n"
            "We're running it back with Catan because I REFUSE to let\n"
            "Dave's longest road strategy go unchallenged.\n\n"
            "Current roster: me, Dave, Meg, you (hopefully).\n"
            "I'll grab beers and chips.\n\n"
            "Also I got the Seafarers expansion if people wanna try it.\n\n"
            "Kev"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.75,
        "age_range": (0, 6),
    },
    # 8 - Travel photos
    {
        "sender_name": "Megan Reyes",
        "sender_email": "meganreyes.sf@gmail.com",
        "subject": "Japan photos are finally up!",
        "body_plain": (
            "Alex!\n\n"
            "I finally sorted through all 2,000 photos from Japan (lol) and\n"
            "made a Google Photos album. The Kyoto temples are unreal.\n\n"
            "Album link: https://photos.google.com\n\n"
            "The food section alone is like 200 photos. That ramen spot in\n"
            "Shinjuku we found at midnight? Life-changing.\n\n"
            "We NEED to plan a group trip back. Tokyo in cherry blossom\n"
            "season would be incredible.\n\n"
            "Meg"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.6,
        "age_range": (5, 20),
    },
    # 9 - Recipe sharing
    {
        "sender_name": "Lisa Tran",
        "sender_email": "lisatran.music@gmail.com",
        "subject": "That pasta recipe you asked about",
        "body_plain": (
            "Hey! Here's the cacio e pepe recipe I was telling you about.\n\n"
            "Ingredients:\n"
            "- 1 lb spaghetti (or tonnarelli if you can find it)\n"
            "- 2 cups Pecorino Romano, finely grated\n"
            "- 2 tsp fresh cracked black pepper\n"
            "- Pasta water (the secret ingredient)\n\n"
            "The trick is the pasta water emulsion -- you HAVE to add it\n"
            "slowly and toss constantly. No cream, no butter, no shortcuts.\n\n"
            "Let me know how it turns out!\n"
            "Lisa"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.5,
        "age_range": (1, 15),
    },
    # 10 - Movie recommendations
    {
        "sender_name": "David Park",
        "sender_email": "dpark.david@gmail.com",
        "subject": "You HAVE to watch this movie",
        "body_plain": (
            "Alex, stop whatever you're doing and go watch Past Lives.\n"
            "I know you said you don't watch romance movies but this\n"
            "is different. Trust me.\n\n"
            "It's on A24's streaming thing. Like 2 hours.\n"
            "Celine Song directed it.\n\n"
            "I literally sat in silence for 20 minutes after it ended.\n"
            "We need to discuss.\n\n"
            "Dave"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.5,
        "age_range": (0, 10),
    },
    # 11 - Old college friend catching up
    {
        "sender_name": "Jason Morales",
        "sender_email": "jmorales.cal@gmail.com",
        "subject": "Long time no talk! Saw your company in TechCrunch",
        "body_plain": (
            "Alex!! Dude!!\n\n"
            "I just saw NexusAI on TechCrunch -- that's YOUR company right??\n"
            "Congrats man, that's insane. I remember when you were sketching\n"
            "out the idea on napkins at Caffe Strada.\n\n"
            "I'm still at Deloitte in Chicago but honestly getting the itch\n"
            "to do something different. Would love to catch up sometime --\n"
            "you free for a call this week or next?\n\n"
            "So proud of you bro.\n\n"
            "Jason"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.6,
        "age_range": (3, 20),
    },
    # 12 - Parents asking about visit
    {
        "sender_name": "Linda Chen",
        "sender_email": "lindachen.mom@gmail.com",
        "subject": "When are you coming home?? Dad wants to know",
        "body_plain": (
            "Hi sweetheart,\n\n"
            "Dad and I were talking and we realized it's been FOUR months\n"
            "since you visited. That's too long.\n\n"
            "Easter is April 20 this year. Can you come home for the\n"
            "weekend? Your sister will be here too. Dad said he'll make\n"
            "his famous braised pork belly.\n\n"
            "Also please eat more vegetables. I worry about you.\n\n"
            "Love,\n"
            "Mom"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.85,
        "age_range": (1, 8),
    },
    # 13 - Sibling sharing news
    {
        "sender_name": "Emily Chen",
        "sender_email": "emilychen.ec@gmail.com",
        "subject": "Guess what!!!",
        "body_plain": (
            "ALEX\n\n"
            "I GOT INTO STANFORD LAW.\n\n"
            "I'm literally shaking right now. The acceptance email came\n"
            "in like 10 minutes ago. I haven't even told mom and dad yet,\n"
            "you're the first person I'm telling.\n\n"
            "Can you call me tonight?? I have so many questions about\n"
            "moving to the Bay Area. You're basically my local expert now.\n\n"
            "AHHHHH\n"
            "Em"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.95,
        "age_range": (0, 5),
    },
    # 14 - Running club invite
    {
        "sender_name": "Nina Gupta",
        "sender_email": "ninagupta.ny@gmail.com",
        "subject": "Join our half marathon training group?",
        "body_plain": (
            "Hey Alex,\n\n"
            "A few of us are training for the SF Half Marathon in July.\n"
            "We run Tues/Thurs mornings at 6:30am from the Embarcadero\n"
            "and do a long run on Saturdays.\n\n"
            "Current pace is around 9:00-9:30/mile for easy runs.\n"
            "We're using the Hal Higdon training plan.\n\n"
            "No pressure but it's way more fun with a group! And we\n"
            "always get coffee after the Saturday run.\n\n"
            "Nina"
        ),
        "body_html": "",
        "category": "CATEGORY_SOCIAL",
        "is_read_probability": 0.4,
        "age_range": (2, 14),
    },
    # 15 - Favor / pet sitting
    {
        "sender_name": "Ryan Nakamura",
        "sender_email": "rnakamura22@gmail.com",
        "subject": "Huge favor -- can you watch Luna for a weekend?",
        "body_plain": (
            "Hey Alex,\n\n"
            "Super random ask but Jess and I are going to Tahoe March 28-30\n"
            "and our usual dog sitter just bailed. Any chance you could\n"
            "watch Luna for the weekend?\n\n"
            "She's super low maintenance -- just two walks a day and she\n"
            "sleeps like 18 hours. I'll drop off her food and everything.\n\n"
            "Totally understand if you can't but you're her favorite\n"
            "human after us so figured I'd ask.\n\n"
            "Ryan"
        ),
        "body_html": "",
        "category": "CATEGORY_PERSONAL",
        "is_read_probability": 0.7,
        "age_range": (0, 10),
    },
]
