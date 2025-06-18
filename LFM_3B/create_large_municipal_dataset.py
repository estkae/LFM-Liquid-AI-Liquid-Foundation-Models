#!/usr/bin/env python3
"""
Create a much larger German municipal training dataset with 500+ examples
"""

import json
import random


def create_large_municipal_dataset(output_file: str = "large_municipal_training_data.jsonl"):
    """Create extensive training data with many variations"""
    
    # Base templates for each department
    templates = {
        "einwohnermeldeamt": [
            {
                "questions": [
                    "Ich möchte meinen Wohnsitz ummelden. Was muss ich tun?",
                    "Wie melde ich meinen Umzug an?",
                    "Welche Unterlagen brauche ich für die Ummeldung?",
                    "Wo kann ich mich ummelden?",
                    "Ich bin umgezogen. Was muss ich beachten?",
                    "Wie lange habe ich Zeit für die Ummeldung?",
                    "Was kostet die Ummeldung?",
                    "Kann ich mich online ummelden?",
                    "Ich ziehe ins Ausland. Muss ich mich abmelden?",
                    "Wie bekomme ich eine Meldebescheinigung?"
                ],
                "answers": [
                    "Für die Ummeldung müssen Sie persönlich beim Einwohnermeldeamt erscheinen. Bringen Sie Personalausweis und Wohnungsgeberbestätigung mit. Die Ummeldung muss innerhalb von 14 Tagen erfolgen.",
                    "Bringen Sie Ihren Personalausweis oder Reisepass sowie die Wohnungsgeberbestätigung vom Vermieter mit. Die Anmeldung ist kostenfrei.",
                    "Sie benötigen: Personalausweis oder Reisepass, Wohnungsgeberbestätigung vom Vermieter oder Eigentümer. Bei Minderjährigen zusätzlich die Geburtsurkunde.",
                    "Die Ummeldung erfolgt im Einwohnermeldeamt Ihres neuen Wohnortes. Termine können online vereinbart werden.",
                    "Melden Sie sich innerhalb von 14 Tagen nach dem Umzug um. Vergessen Sie nicht, auch Ihre Kfz-Zulassung und Krankenversicherung zu informieren.",
                    "Die Ummeldung muss innerhalb von 14 Tagen nach dem Einzug erfolgen. Bei Versäumung droht ein Bußgeld von 5-50 Euro.",
                    "Die An- und Ummeldung ist kostenfrei. Nur eine Meldebescheinigung kostet 5 Euro.",
                    "Eine Online-Ummeldung ist derzeit nicht möglich. Sie müssen persönlich im Einwohnermeldeamt erscheinen.",
                    "Ja, bei Wegzug ins Ausland müssen Sie sich abmelden. Die Abmeldung ist kostenfrei und Sie erhalten eine Abmeldebescheinigung.",
                    "Eine Meldebescheinigung erhalten Sie gegen 5 Euro Gebühr im Einwohnermeldeamt. Personalausweis mitbringen."
                ]
            }
        ],
        "bauamt": [
            {
                "questions": [
                    "Brauche ich eine Baugenehmigung für einen Wintergarten?",
                    "Wie beantrage ich eine Baugenehmigung?",
                    "Was kostet eine Baugenehmigung?",
                    "Wie lange dauert die Bearbeitung einer Baugenehmigung?",
                    "Kann ich ohne Genehmigung ein Gartenhaus bauen?",
                    "Welche Unterlagen brauche ich für den Bauantrag?",
                    "Wer darf Bauanträge einreichen?",
                    "Kann ich meinen Balkon verglansen?",
                    "Brauche ich eine Genehmigung für eine Terrasse?",
                    "Was ist bei einer Grenzbebauung zu beachten?"
                ],
                "answers": [
                    "Für einen Wintergarten benötigen Sie in der Regel eine Baugenehmigung. Reichen Sie Bauzeichnungen, Lageplan und Baubeschreibung ein.",
                    "Reichen Sie den Bauantrag mit vollständigen Unterlagen beim Bauamt ein. Die Bearbeitungszeit beträgt etwa 3 Monate.",
                    "Die Gebühren richten sich nach den Baukosten. Als Faustregel gilt: 0,5% bis 1% der Baukosten.",
                    "Die normale Bearbeitungszeit beträgt 3 Monate. Bei vollständigen Unterlagen kann es auch schneller gehen.",
                    "Gartenhäuser bis 30 Kubikmeter sind oft genehmigungsfrei, müssen aber baurechtliche Vorschriften einhalten.",
                    "Sie benötigen: Bauzeichnungen, Lageplan, Baubeschreibung, statische Berechnungen und einen Nachweis der Erschließung.",
                    "Bauanträge dürfen nur von bauvorlageberechtigten Personen eingereicht werden, z.B. Architekten oder Ingenieure.",
                    "Eine Balkonverglasung ist meist genehmigungspflichtig. Informieren Sie sich beim Bauamt über die Voraussetzungen.",
                    "Für größere Terrassen kann eine Baugenehmigung erforderlich sein. Fragen Sie beim Bauamt nach.",
                    "Bei Grenzbebauung sind die Abstandsregeln zu beachten. Oft ist eine Zustimmung des Nachbarn erforderlich."
                ]
            }
        ],
        "standesamt": [
            {
                "questions": [
                    "Wie beantrage ich eine Geburtsurkunde?",
                    "Was kostet eine Geburtsurkunde?",
                    "Welche Unterlagen brauche ich für die Eheschließung?",
                    "Wie lange dauert es, eine Sterbeurkunde zu bekommen?",
                    "Kann ich online eine Geburtsurkunde beantragen?",
                    "Wo bekomme ich eine beglaubigte Kopie meiner Heiratsurkunde?",
                    "Wie ändere ich meinen Namen nach der Heirat?",
                    "Was brauche ich für die Anmeldung zur Eheschließung?",
                    "Kann ich im Ausland geheiratet haben anerkennen lassen?",
                    "Wie beantrage ich ein Lebenspartnerschaftszeugnis?"
                ],
                "answers": [
                    "Geburtsurkunden beantragen Sie beim Standesamt des Geburtsortes. Möglich ist dies persönlich, schriftlich oder online.",
                    "Eine Geburtsurkunde kostet 12 Euro. Jede weitere Ausfertigung kostet ebenfalls 12 Euro.",
                    "Sie benötigen: Personalausweis, beglaubigte Abschrift aus dem Geburtenregister (max. 6 Monate alt), Aufenthaltsbescheinigung.",
                    "Sterbeurkunden werden sofort ausgestellt, wenn alle Unterlagen vorliegen. Die Gebühr beträgt 12 Euro.",
                    "Ja, in vielen Gemeinden können Sie Geburtsurkunden online beantragen. Prüfen Sie das Online-Portal Ihrer Stadt.",
                    "Heiratsurkunden erhalten Sie beim Standesamt des Heiratsortes. Die Gebühr beträgt 12 Euro pro Urkunde.",
                    "Nach der Heirat können Sie den Namen beim Einwohnermeldeamt ändern lassen. Bringen Sie die Heiratsurkunde mit.",
                    "Zur Anmeldung der Eheschließung benötigen Sie Personalausweis, Geburtsurkunde und Aufenthaltsbescheinigung.",
                    "Ausländische Eheschließungen können beim Standesamt nachbeurkundet werden. Bringen Sie alle Originaldokumente mit.",
                    "Lebenspartnerschaftszeugnisse erhalten Sie beim Standesamt, das die Lebenspartnerschaft begründet hat."
                ]
            }
        ],
        "ordnungsamt": [
            {
                "questions": [
                    "Ich möchte mich über Lärmbelästigung beschweren.",
                    "Wie beantrage ich eine Sondernutzungserlaubnis?",
                    "Mein Nachbar parkt vor meiner Einfahrt. Was kann ich tun?",
                    "Wo kann ich Falschparker melden?",
                    "Brauche ich eine Genehmigung für ein Straßenfest?",
                    "Was kostet ein Bewohnerparkausweis?",
                    "Wie melde ich wilden Müll?",
                    "Brauche ich eine Erlaubnis für einen Umzugswagen?",
                    "Was kann ich gegen Hundekot auf dem Gehweg tun?",
                    "Wie beantrage ich eine Ausnahmegenehmigung zum Befahren der Fußgängerzone?"
                ],
                "answers": [
                    "Bei Lärmbelästigung wenden Sie sich an das Ordnungsamt. Dokumentieren Sie die Störungen mit Datum und Uhrzeit.",
                    "Sondernutzungserlaubnisse beantragen Sie beim Ordnungsamt. Der Antrag muss mindestens 5 Werktage vorher gestellt werden.",
                    "Das Zuparken von Einfahrten ist eine Ordnungswidrigkeit. Das Ordnungsamt kann Bußgelder verhängen und abschleppen lassen.",
                    "Falschparker können Sie telefonisch oder online beim Ordnungsamt melden. Halten Sie das Kennzeichen bereit.",
                    "Für Straßenfeste benötigen Sie eine Sondernutzungserlaubnis. Beantragen Sie diese frühzeitig beim Ordnungsamt.",
                    "Ein Bewohnerparkausweis kostet 30 Euro pro Jahr. Den Antrag stellen Sie beim Ordnungsamt oder online.",
                    "Wilden Müll können Sie dem Ordnungsamt telefonisch oder per App melden. Die Beseitigung wird veranlasst.",
                    "Für Umzugswagen auf öffentlichen Straßen benötigen Sie eine Sondernutzungserlaubnis. Gebühr: ca. 30-50 Euro pro Tag.",
                    "Hundekot auf Gehwegen ist ordnungswidrig. Melden Sie Verstöße dem Ordnungsamt mit Ort und Zeit.",
                    "Ausnahmegenehmigungen für die Fußgängerzone beantragen Sie beim Ordnungsamt. Begründen Sie den Antrag ausführlich."
                ]
            }
        ],
        "stadtkasse": [
            {
                "questions": [
                    "Wie kann ich meine Grundsteuer bezahlen?",
                    "Ich habe eine Mahnung erhalten. Was soll ich tun?",
                    "Wo finde ich meine Steuernummer?",
                    "Kann ich meine Steuerschuld in Raten zahlen?",
                    "Was passiert, wenn ich nicht zahle?",
                    "Wie beantrage ich Stundung?",
                    "Wo kann ich bar bezahlen?",
                    "Wie richte ich ein SEPA-Lastschriftverfahren ein?",
                    "Wann sind Steuern fällig?",
                    "Kann ich eine Zahlungsbestätigung erhalten?"
                ],
                "answers": [
                    "Die Grundsteuer können Sie per Überweisung, SEPA-Lastschrift oder bar in der Stadtkasse bezahlen.",
                    "Prüfen Sie, ob die Zahlung bereits erfolgt ist. Falls nicht, zahlen Sie umgehend. Bei Problemen kontaktieren Sie die Stadtkasse.",
                    "Ihre Steuernummer finden Sie auf dem Grundsteuerbescheid oder anderen Bescheiden der Stadt.",
                    "Bei Zahlungsschwierigkeiten können Sie eine Ratenzahlung beantragen. Wenden Sie sich an die Stadtkasse.",
                    "Bei Nichtzahlung erfolgen Mahnungen und später Vollstreckungsmaßnahmen. Zinsen und Gebühren kommen hinzu.",
                    "Stundung können Sie schriftlich beantragen. Begründen Sie Ihre finanzielle Notlage ausführlich.",
                    "Barzahlungen sind zu den Öffnungszeiten in der Stadtkasse möglich: Mo-Mi 8-16 Uhr, Do 8-18 Uhr, Fr 8-12 Uhr.",
                    "Das SEPA-Lastschriftverfahren richten Sie mit dem entsprechenden Formular bei der Stadtkasse ein.",
                    "Grundsteuer ist vierteljährlich fällig: 15. Februar, 15. Mai, 15. August, 15. November.",
                    "Zahlungsbestätigungen erhalten Sie auf Antrag bei der Stadtkasse. Die Gebühr beträgt 5 Euro."
                ]
            }
        ],
        "sozialamt": [
            {
                "questions": [
                    "Wie beantrage ich Wohngeld?",
                    "Wer hat Anspruch auf Grundsicherung?",
                    "Gibt es Hilfe für die Wohnungserstausstattung?",
                    "Wie beantrage ich Hilfe zur Pflege?",
                    "Was ist das Bildungs- und Teilhabepaket?",
                    "Wo bekomme ich Schuldnerberatung?",
                    "Wie beantrage ich Sozialhilfe?",
                    "Gibt es Unterstützung für Alleinerziehende?",
                    "Was ist Eingliederungshilfe?",
                    "Wie bekomme ich einen Sozialpass?"
                ],
                "answers": [
                    "Wohngeld beantragen Sie mit dem Antragsformular beim Sozialamt. Benötigt werden Einkommensnachweise und Mietvertrag.",
                    "Grundsicherung erhalten Personen ab 65 Jahren oder mit voller Erwerbsminderung, deren Einkommen nicht ausreicht.",
                    "Bei Bedürftigkeit gewährt das Sozialamt Beihilfen für Erstausstattung. Antrag mit Kostenvoranschlägen stellen.",
                    "Hilfe zur Pflege beantragen Sie beim Sozialamt. Voraussetzung ist Pflegebedürftigkeit und unzureichendes Einkommen.",
                    "Das Bildungspaket unterstützt Kinder aus bedürftigen Familien bei Schulausflügen, Lernförderung und Vereinsbeiträgen.",
                    "Kostenlose Schuldnerberatung bieten anerkannte Beratungsstellen. Termine vereinbaren Sie über das Sozialamt.",
                    "Sozialhilfe beantragen Sie beim örtlichen Sozialamt. Bringen Sie alle Unterlagen zu Einkommen und Vermögen mit.",
                    "Alleinerziehende können Mehrbedarf bei der Grundsicherung und Unterstützung bei der Kinderbetreuung erhalten.",
                    "Eingliederungshilfe unterstützt Menschen mit Behinderungen bei der Teilhabe am gesellschaftlichen Leben.",
                    "Den Sozialpass erhalten bedürftige Personen beim Sozialamt. Er ermöglicht Vergünstigungen bei städtischen Einrichtungen."
                ]
            }
        ],
        "jugendamt": [
            {
                "questions": [
                    "Wie beantrage ich einen Kita-Platz?",
                    "Welche Unterlagen brauche ich für Elterngeld?",
                    "Gibt es Ferienbetreuung für Schulkinder?",
                    "Wie bekomme ich Unterstützung bei der Kinderbetreuung?",
                    "Was macht die Erziehungsberatung?",
                    "Wie beantrage ich Kindertagespflege?",
                    "Gibt es finanzielle Hilfe für Klassenfahrten?",
                    "Was ist das Kinderschutzgesetz?",
                    "Wie bekomme ich einen Betreuungsplatz für unter 3-Jährige?",
                    "Was kostet die Kinderbetreuung?"
                ],
                "answers": [
                    "Kita-Plätze beantragen Sie über das Online-Portal oder direkt beim Jugendamt. Anmeldung ab Geburt möglich.",
                    "Für Elterngeld benötigen Sie: Geburtsurkunde, Einkommensnachweise, Krankenkassenbescheinigung, Arbeitgeberbescheinigung.",
                    "Das Jugendamt organisiert Ferienbetreuung für Schulkinder. Anmeldung online, Plätze sind begrenzt.",
                    "Bei Kinderbetreuung unterstützt das Jugendamt mit Beratung und vermittelt Betreuungsplätze und Tagesmütter.",
                    "Die Erziehungsberatung hilft bei Problemen in der Familie und Erziehungsfragen. Beratung ist kostenfrei und vertraulich.",
                    "Kindertagespflege vermittelt das Jugendamt. Tagesmütter werden überprüft und qualifiziert.",
                    "Über das Bildungspaket können bedürftige Familien Unterstützung für Klassenfahrten erhalten.",
                    "Das Kinderschutzgesetz stärkt den Schutz von Kindern vor Vernachlässigung und Gewalt.",
                    "Für unter 3-Jährige gibt es einen Rechtsanspruch auf Betreuung. Beantragung über das Portal der Stadt.",
                    "Die Kosten richten sich nach Einkommen und Betreuungszeit. Gering verdienende Familien zahlen weniger oder nichts."
                ]
            }
        ],
        "general": [
            {
                "questions": [
                    "Wo finde ich Formulare zum Download?",
                    "Wie sind die Öffnungszeiten des Rathauses?",
                    "Kann ich online einen Termin vereinbaren?",
                    "Wo ist das Bürgerbüro?",
                    "Wie erreiche ich die Stadtverwaltung?",
                    "Gibt es kostenloses WLAN im Rathaus?",
                    "Wo kann ich meinen Personalausweis verlängern?",
                    "Ist das Rathaus barrierefrei?",
                    "Wo finde ich Informationen zu Veranstaltungen?",
                    "Wie kann ich der Stadt eine Anregung mitteilen?"
                ],
                "answers": [
                    "Alle Formulare finden Sie auf der Webseite unter 'Service/Formulare' oder im Bürgerbüro.",
                    "Öffnungszeiten: Mo-Mi 8-16 Uhr, Do 8-18 Uhr, Fr 8-12 Uhr. Termine außerhalb nach Vereinbarung.",
                    "Termine können Sie online über das Terminbuchungssystem der Stadt vereinbaren.",
                    "Das Bürgerbüro befindet sich im Erdgeschoss des Rathauses, Eingang Hauptstraße.",
                    "Telefon: 0123-456789, E-Mail: info@stadt.de. Persönlich zu den Öffnungszeiten.",
                    "Ja, kostenloses WLAN 'Stadt-WLAN' ist im gesamten Rathaus verfügbar.",
                    "Personalausweise werden im Bürgerbüro verlängert. Termine online buchbar.",
                    "Ja, das Rathaus ist vollständig barrierefrei. Aufzug und behindertengerechte Toiletten vorhanden.",
                    "Veranstaltungshinweise finden Sie auf der städtischen Webseite und im monatlichen Stadtmagazin.",
                    "Anregungen können Sie per E-Mail, Brief oder über das Online-Formular 'Bürgerwünsche' einreichen."
                ]
            }
        ]
    }
    
    # Generate all combinations
    all_examples = []
    
    for dept, dept_data in templates.items():
        for template_group in dept_data:
            questions = template_group["questions"]
            answers = template_group["answers"]
            
            # Pair each question with each answer (creates realistic variations)
            for i, question in enumerate(questions):
                # Primary answer
                primary_answer = answers[i] if i < len(answers) else answers[0]
                
                # Different formats
                formats = [
                    f"Frage: {question}\nAntwort: {primary_answer}",
                    f"Q: {question}\nA: {primary_answer}",
                    f"Bürger: {question}\nSachbearbeiter: {primary_answer}",
                    f"Kunde: {question}\nMitarbeiter: {primary_answer}",
                    f"{question}\n{primary_answer}",
                    f"Anfrage: {question}\nAuskunft: {primary_answer}"
                ]
                
                for format_text in formats:
                    all_examples.append({
                        "text": format_text,
                        "metadata": {
                            "department": dept,
                            "question_type": "standard"
                        }
                    })
                
                # Add some cross-department answers for variety
                if random.random() < 0.3:  # 30% chance
                    other_answers = [ans for dept_name, dept_info in templates.items() 
                                   if dept_name != dept 
                                   for tg in dept_info for ans in tg["answers"]]
                    if other_answers:
                        cross_answer = random.choice(other_answers)
                        all_examples.append({
                            "text": f"Frage: {question}\nAntwort: {cross_answer}",
                            "metadata": {
                                "department": "mixed",
                                "question_type": "cross_reference"
                            }
                        })
    
    # Add variations with incomplete questions/prompts
    prompt_starters = [
        "Ich brauche Hilfe bei",
        "Können Sie mir sagen",
        "Wo kann ich",
        "Wie funktioniert",
        "Was muss ich tun für",
        "Ich möchte gerne",
        "Haben Sie Informationen zu"
    ]
    
    for starter in prompt_starters:
        for dept, dept_data in templates.items():
            for template_group in dept_data:
                for question in template_group["questions"][:3]:  # Only first 3 per department
                    modified_q = f"{starter} {question.lower()}"
                    answer = template_group["answers"][0]
                    all_examples.append({
                        "text": f"Frage: {modified_q}\nAntwort: {answer}",
                        "metadata": {
                            "department": dept,
                            "question_type": "prompt_variation"
                        }
                    })
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(all_examples)} training examples in {output_file}")
    
    # Print statistics
    dept_counts = {}
    type_counts = {}
    for ex in all_examples:
        dept = ex["metadata"]["department"]
        qtype = ex["metadata"]["question_type"]
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    print(f"\n📊 Training data statistics:")
    print(f"Total examples: {len(all_examples)}")
    print("\nBy department:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept}: {count} examples")
    
    print("\nBy question type:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count} examples")
    
    return output_file


if __name__ == "__main__":
    create_large_municipal_dataset()