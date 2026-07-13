from app.services.categorizer import CategorizationService

PAUL_GUIDE_DOG_TX = (
    "Hello everyone, this is Paul. Last week I was out walking with my guide dog Rocco, "
    "something we do most days. I was listening through my Meta Ray-Ban glasses. "
    "There's an app called Orion, an AI assistant designed specifically for visually impaired "
    "users. Independence isn't about doing everything on your own. It's about having the freedom "
    "to choose how you experience the world. The sun was warm and trees and a lake were described "
    "through the glasses."
)

MING_CHAN_TX = (
    "Bai Ming Chan, a graphic designer from Chadwell Heath, has won the Second Wildlife Award "
    "for his video titled 'The Elegant in Water,' which showcases the lives of great crested "
    "grebes over a two-year period. Ming, originally from Hong Kong, follows the birds from "
    "hunting to mating and raising chicks in his local country park. This award comes just "
    "two months after he won the Essex Wildlife Trust's annual photography competition for "
    "his photo 'Run Rabbit Run.' Ming hopes to capture more wildlife as part of the rewilding "
    "East London scheme."
)


def test_guide_dog_smart_glasses_not_wildlife_or_generic_technology():
    svc = CategorizationService()
    tags, cats = svc._apply_editorial_rules(
        PAUL_GUIDE_DOG_TX,
        ["#wildlife", "#technology", "#animals", "#podcast"],
        ["Accessibility", "Wildlife", "Technology"],
        8,
    )
    assert "Wildlife" not in cats
    assert "Technology" not in cats
    assert "#wildlife" not in tags
    assert "#technology" not in tags
    assert "Personal lived experience" in cats
    assert "#accessibility" in tags
    assert "#guidedogs" in tags or "#assistivetechnology" in tags


def test_wildlife_award_story_not_veterinary():
    svc = CategorizationService()
    tags, cats = svc._apply_editorial_rules(
        MING_CHAN_TX,
        ["#podcast"],
        ["Veterinary", "Podcast"],
        8,
    )
    assert "Veterinary" not in cats
    assert "Wildlife" in cats
    assert "News" in cats or "Photography" in cats
    assert "#wildlife" in tags


DARENT_RIVER_TX = (
    "Sevenoaks Town Council has allocated £500 to the Darent River Preservation Society "
    "(DRIPS) to fund a project aimed at improving the flow of the Darent River, one of the "
    "most over-extracted chalk streams in the UK. The charity is collaborating with water "
    "companies to explore solutions that do not involve costly infrastructure projects, "
    "such as building a new reservoir or sewage treatment plant."
)


def test_river_council_story_not_assistive_taxonomy():
    svc = CategorizationService()
    tags, cats = svc._sanitize_categorization_labels(
        DARENT_RIVER_TX,
        ["Accessibility > Visual impairment > Assistive technology > Smart glasses"],
        ["Accessibility > Visual impairment > Assistive technology > Smart glasses"],
    )
    assert not any(" > " in t for t in tags)
    assert not any(" > " in c for c in cats)
    assert "Smart glasses" not in cats
    tags, cats = svc._apply_editorial_rules(DARENT_RIVER_TX, tags, cats, 8)
    assert any(c in cats for c in ("Environment", "Community", "Charity", "News"))
    assert any(t in tags for t in ("#environment", "#community", "#charity", "#water"))
    assert not any("smart" in c.lower() for c in cats)
    assert not any(" > " in c for c in cats)


def test_finalize_categories_blocks_veterinary_for_wildlife_media():
    svc = CategorizationService()
    zs = {
        "Veterinary": 0.91,
        "Wildlife": 0.4,
        "News": 0.35,
        "Podcast": 0.5,
    }
    cats = svc._finalize_categories(MING_CHAN_TX, ["Veterinary"], zs, max_categories=3)
    assert "Veterinary" not in cats
    assert "Wildlife" in cats
