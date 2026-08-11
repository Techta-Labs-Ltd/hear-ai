import json

from hear.models.database import CategoryTrainingExample, HarmKeyword, SessionLocal

HARM_KEYWORDS = [
    "i will kill you", "i'm going to kill you", "i am going to kill you",
    "gonna kill you", "gon kill you", "imma kill you", "i'll kill you",
    "i will murder you", "i will end you", "i will hurt you",
    "i am going to beat", "i'm going to beat", "gonna beat her up", "gonna beat him up",
    "beat her up", "beat him up", "beat you up", "beat them up",
    "beat her badly", "beat him badly", "beat her very badly", "beat him very badly",
    "i'll beat you", "i will beat you", "imma beat you", "smash her face",
    "smash his face", "punch her", "punch him", "kick her head", "kick his head",
    "you're dead", "you are dead", "you're dead to me",
    "i'll shoot you", "i will shoot you", "i'll stab you", "i will stab you",
    "i'll blow your head off", "i'll put a bullet in you",
    "i want you dead", "i hope you die",
    "spin the block", "up the score", "catch a body", "caught a body",
    "drop a body", "wet him up", "wet her up", "splash him", "splash her",
    "poke him", "poke her", "shank him", "shank her", "shank them",
    "ride out on", "back out the strap", "back out the tool",
    "link man with the strap", "up the stick", "do him dirty",
    "score on sight", "put him in the dirt",
    "slide on", "leave him leaking", "leave her leaking", "leave man leaking",
    "stomp him out", "stomp her out", "rush him", "rush her",
    "spin his block", "spin her block", "skeng", "no lacking",
    "cap him", "clap him", "blick him", "smoke him", "smoke her", "smoke them",
    "catch him slipping", "catch her slipping", "run up on him", "run up on her",
    "squeeze the trigger", "pull the trigger on",
    "put him six feet", "put her six feet", "send him to god",
    "drop him", "drop her", "body bag him", "body bag her",
    "get smoked", "get merked", "get rocked", "get domed",
    "put hands on", "beat him down", "beat her down",
    "bust at", "bust shots at", "send shots", "let it bang",
    "bust a cap", "dump on", "chop him down", "chop her down",
    "air him out", "air her out", "hit the lick", "rob him", "rob her",
    "with my strap", "strapped up", "glizzy", "draco",
    "mac-10", "uzi", "ar-15", "banana clip", "extended clip", "drum mag",
    "glock with a switch", "auto switch",
    "kill all", "death to", "gas the", "exterminate the",
    "white power", "white supremacy", "nigger", "nigga die",
    "kike", "spic", "chink", "sand nigger", "raghead",
    "bomb the mosque", "bomb the church", "blow up the school",
    "jihad against", "holy war against",
    "child porn", "kiddie porn", "cp link", "rape a child",
    "molest a child", "underaged naked",
    "kill myself", "killing myself", "want to die", "going to end it",
    "slit my wrists", "hang myself", "suicide pact",
    "overdose on purpose", "want to overdose",
    "allahu akbar bomb", "suicide bomb", "blow myself up",
    "school shooting", "mass shooting", "shoot up the", "bomb threat",
    "pipe bomb", "nail bomb",
]

SAFE_EXAMPLES = [
    "the weather today is sunny and warm",
    "i went to the grocery store to buy milk and bread",
    "the meeting is scheduled for three pm tomorrow",
    "she is reading a book about ancient history",
    "we are planning a trip to the mountains next weekend",
    "the new restaurant downtown has great reviews",
    "he fixed the broken sink in the bathroom",
    "they are watching a documentary about ocean life",
    "the children are playing soccer in the park",
    "i need to finish this report by friday",
    "she learned to play the piano when she was six",
    "the train arrives at platform four in ten minutes",
    "we planted tomatoes and basil in the garden",
    "he is studying computer science at the university",
    "the concert was amazing, the band played for three hours",
    "i adopted a rescue dog from the shelter last month",
    "she baked chocolate chip cookies for the bake sale",
    "the new software update includes several bug fixes",
    "they went hiking in the national park yesterday",
    "i am learning spanish using a language app",
    "the coffee shop on the corner makes excellent lattes",
    "he volunteers at the community center every saturday",
    "the art gallery is featuring local artists this month",
    "we celebrated her birthday with a small dinner party",
    "the bridge will be closed for repairs this weekend",
    "she started a podcast about sustainable living",
    "i forgot my umbrella and got caught in the rain",
    "the museum has a new exhibit on egyptian artifacts",
    "he is training for a marathon in the fall",
    "the library is hosting a free coding workshop",
]


def seed() -> dict:
    db = SessionLocal()
    inserted = {"harm_seed": 0, "safe_seed": 0, "db_harm_seed": 0}
    try:
        existing_texts = {
            row.text for row in db.query(CategoryTrainingExample.text)
            .filter(CategoryTrainingExample.source == "harm_seed").all()
        }

        for kw in HARM_KEYWORDS:
            kw = kw.strip().lower()
            if not kw or kw in existing_texts:
                continue
            db.add(CategoryTrainingExample(
                source="harm_seed",
                event_type="seed_harm",
                text=kw,
                label="harmful",
            ))
            existing_texts.add(kw)
            inserted["harm_seed"] += 1

        for kw_safe in SAFE_EXAMPLES:
            if kw_safe in existing_texts:
                continue
            db.add(CategoryTrainingExample(
                source="harm_seed",
                event_type="seed_safe",
                text=kw_safe,
                label="safe",
            ))
            existing_texts.add(kw_safe)
            inserted["safe_seed"] += 1

        for row in db.query(HarmKeyword).filter(HarmKeyword.kind == "harm").all():
            kw = row.keyword.strip().lower()
            if not kw or kw in existing_texts:
                continue
            db.add(CategoryTrainingExample(
                source="harm_seed",
                event_type="seed_harm_db",
                text=kw,
                label="harmful",
            ))
            existing_texts.add(kw)
            inserted["db_harm_seed"] += 1

        db.commit()
    finally:
        db.close()
    return inserted


if __name__ == "__main__":
    print(json.dumps(seed(), indent=2))
