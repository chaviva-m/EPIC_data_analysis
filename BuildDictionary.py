from statistics import mean, stdev
from scipy.stats import ttest_ind
from pickle import dump
import os
import re
import numpy as np
from collections import Counter
from operator import itemgetter
import csv

VIDEO_DIR = 'data\\VideoFiles\\'
VIDEO_BL_DIR = 'data\\VideoBaselineFiles\\'


DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'
INVALID_LIST = [999,37,43]


def createParticipants():
    emotions = {
        0: "happy",
        1: "sad",
        2: "neutral"
    }
    details = {}
    participants = {}
    with open(DATA_DIR + 'Participant.csv') as participantFile:
        for line in participantFile.readlines():
            details = {}
            l = [x.strip() for x in line.split(',')]
            if (l[0] == '0'):
                continue
            if (l[0] == 'NULL'):
                return participants
            #read details from participant data table
            identification = int(l[0][-3:])
            details['emotion'] = (emotions.get(int(l[2])), int(l[2]))
            details['videoBaseline'] = l[3]
            details['videoBaselineLabeled'] = l[4]
            details['videoBaselineData'] = l[5]
            details['video'] = l[6]
            details['videoLabeled'] = l[7]
            details['videoData'] = l[8]
            details['audioBaseline'] = l[9]
            details['audioBaselineData'] = l[10]
            details['audio'] = l[11]
            details['audioData'] = l[12]
            details['writingTime'] = float(l[13])
            details['ultimatumOffer'] = float(l[14])
            details['ultimatumOfferPercent'] = float(l[15])
            details['ultimatumInstructionRT'] = float(l[16])
            details['ultimatumDMrt'] = float(l[17])
            details['trustOffer'] = float(l[18])
            details['trustOfferPercent'] = float(l[19])
            details['trustInstructionRT'] = float(l[20])
            details['trustDMrt'] = float(l[21])
            selfReport = {}
            videoBLFreq = {}
            videoFreq = {}
            audioBL = {}
            audio = {}
            details['selfReport'] = selfReport
            details['videoBLFreq'] = videoBLFreq
            details['videoFreq'] = videoFreq
            details['audioBL'] = audioBL
            details['audio'] = audio
            participants[identification] = details

    return participants


def getSelfReportData(participantDict):
    #get info from self-report data table
    emotions = {
        1: "apathy",
        2: "sadness",
        3: "calm",
        4: "amusement",
        5: "grief",
        6: "happiness"
    }
    emotion = 1
    with open(DATA_DIR + 'SelfReport.csv') as selfReportFile:
        for line in selfReportFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['selfReport'][emotions.get(emotion)] = int(l[3])
            emotion += 1
            if (emotion == 7):
                emotion = 1
    return


def getVideoData(participantDict):
    emotions = {
        1: "angry",
        2: "disgust",
        3: "fear",
        4: "happy",
        5: "sad",
        6: "surprise",
        7: "neutral"
    }
    emotion = 1
    with open(DATA_DIR + 'VideoBaseline.csv') as videoBLFile:
        for line in videoBLFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                break
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['videoBLFreq'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 8):
                emotion = 1

    with open(DATA_DIR + 'VideoEmotion.csv') as videoFile:
        for line in videoFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['videoFreq'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 8):
                emotion = 1
    return


def getAudioData(participantDict):
    emotions = {
        1: "neutral",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fear"
    }
    emotion = 1
    with open(DATA_DIR + 'AudioBaseline.csv') as audioBLFile:
        for line in audioBLFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                break
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['audioBL'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 6):
                emotion = 1

    with open(DATA_DIR + 'AudioEmotion.csv') as audioFile:
        for line in audioFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['audio'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 6):
                emotion = 1
    return


def getVideoEmotionProbMean(directory, filename_type, participants, key):
    for filename in os.listdir(directory):
        if re.match(filename_type + '[0-9]+', filename):
            participant_id = int(re.search('[0-9]+', filename).group()[-3:])
            with open(directory + filename, 'r') as f:
                emotions = f.readline().split()[1:]
                emotion_prob = np.array([0.0] * len(emotions))
                num_lines = 0
                for line in f.readlines():
                    num_lines += 1
                    prob = np.array([float(p) for p in line.split()[1:]])
                    emotion_prob += prob
                emotion_prob = emotion_prob / num_lines
            emotion_dict = dict(zip(emotions, emotion_prob))
            participants[participant_id][key] = emotion_dict


def getVideoAboveThresholdEmotionFreq(directory, filename_type, participants, key):
    threshold = 0.5
    for filename in os.listdir(directory):
        if re.match(filename_type + '[0-9]+', filename):
            participant_id = int(re.search('[0-9]+', filename).group()[-3:])
            with open(directory + filename, 'r') as f:
                emotions = f.readline().split()[1:]
                emotions_count = Counter()
                for line in f.readlines():
                    prediction = line.split()[0]
                    probabilities = [float(p) for p in line.split()[1:]]
                    if probabilities[emotions.index(prediction)] >= threshold:
                        emotions_count[prediction] += 1
                total = sum(emotions_count.values())
                emotions_freq = {e:v/total for e,v in emotions_count.items()}
                # add zero frequencies for emotions that never passed threshold
                for e in emotions:
                    if e not in emotions_freq.keys():
                        emotions_freq[e] = 0
                participants[participant_id][key] = emotions_freq


def getVideoEmotionFreqSkipBeginning(directory, filename_type, participants, key):
    n_skip = 100
    for filename in os.listdir(directory):
        if re.match(filename_type + '[0-9]+', filename):
            participant_id = int(re.search('[0-9]+', filename).group()[-3:])
            with open(directory + filename, 'r') as f:
                emotions = f.readline().split()[1:]
                frames = f.readlines()
                frames_relevant = frames[n_skip:]
                frames_predictions = [line.split()[0] for line in frames_relevant]
                emotion_pred, emotion_counts = np.unique(frames_predictions,
                                                         return_counts=True)
                emotions_freq = emotion_counts / len(frames_predictions)
                freq_dict = {e: c for e, c in zip(emotion_pred, emotions_freq)}
                # add zero frequencies for emotions that weren't found
                for e in emotions:
                    if e not in freq_dict.keys():
                        freq_dict[e] = 0
                participants[participant_id][key] = freq_dict


def getVideoFreqTopPercent(directory, filename_type, participants, key):
    percent = 5/100
    for filename in os.listdir(directory):
        if re.match(filename_type + '[0-9]+', filename):
            participant_id = int(re.search('[0-9]+', filename).group()[-3:])
            with open(directory + filename, 'r') as f:
                emotions = f.readline().split()[1:]
                emotions_pred_prob = []
                for line in f.readlines():
                    prediction = line.split()[0]
                    probabilities = [float(p) for p in line.split()[1:]]
                    emotions_pred_prob.append((prediction,probabilities[emotions.index(prediction)]))
                total = len(emotions_pred_prob)
                # sort according to probabilites
                emotions_pred_prob.sort(key=itemgetter(1))
                top_percent = emotions_pred_prob[-round(percent*total):]
                emotions_pred_list = [e[0] for e in top_percent]
                emotion_pred, emotion_counts = np.unique(emotions_pred_list,
                                                         return_counts=True)
                emotions_freq = emotion_counts / len(emotions_pred_list)
                freq_dict = {e: c for e, c in zip(emotion_pred, emotions_freq)}
                # add zero frequencies for emotions that weren't found
                for e in emotions:
                    if e not in freq_dict.keys():
                        freq_dict[e] = 0
                participants[participant_id][key] = freq_dict


def removeInvalidParticipants(participants):
    for invalid in INVALID_LIST:
        try:
            del participants[invalid]
        except KeyError:
            pass



def getPreliminaryQuestionnaireData(participantDict):
    dict = {}
    genders = {
        "נקבה": 0,
        "זכר": 1,
        "אחר": 2
    }
    statuses = {
        "רווק/ה": "single",
        "נשוי/אה": "married",
        "גרוש/ה": "divorced",
        "אלמן/ה": "widowed"
    }
    countries = {
        "ישראל": "Israel",
        "ארצות הברית": "USA",
        "אתיופיה": "Ethiopia",
        "ברית המועצות לשעבר": "USSR",
        "צרפת": "France"
    }
    educations = {
        "עד תיכונית": "high-school",
        "תיכונית (בגרות מלאה)": "Bagrut",
        "אקדמאית- תואר ראשון": "Ba",
        "אקדמאית- תואר שני": "Ms",
        "אקדמאית- תואר שלישי": "Phd"
    }
    yesOrNo = {
        "כן": 1,
        "לא": 0
    }
    economicStates = {
        "נמוך": 1,
        "נמוך-בינוני": 2,
        "בינוני": 3,
        "בינוני-גבוה": 4,
        "גבוה": 5
    }
    religiousAffiliations = {
        "יהודית": "J",
        "מוסלמית": "M",
        "נוצרית": "C",
        "דרוזית": "D"
    }
    agreement = {
        '1 (לא מסכים כלל)': 1,
        '2 (די מתנגד)': 2,
        '3 (לא מסכים ולא מתנגד)': 3,
        '4 (די מסכים)': 4,
        '5 (מסכים מאוד)': 5
    }
    with open(DATA_DIR + 'PreliminaryQuestionnaire.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        titles = next(reader)
        for row in reader:
            if row[0] == '':
                continue

            participant = participantDict.get(int(row[titles.index('מספר נבדק (המספר שפלג נתן לך)')]))
            if participant is None:
                continue
            participant['yearOfBirth'] = int(row[titles.index('שנת לידה')])
            participant['gender'] = int(genders.get(row[titles.index('מגדר')]))
            participant['status'] = statuses.get(row[titles.index('מצב משפחתי')])
            participant['Birthplace'] = countries.get(row[titles.index('ארץ לידה')],"other")
            participant['FatherBirthplace'] = countries.get(row[titles.index('ארץ לידת האב')], "other")
            participant['MotherBirthplace'] = countries.get(row[titles.index('ארץ לידת האם')], "other")
            participant['education'] = educations.get(row[titles.index('השכלה (ניתן לסמן גם אם הינך במהלך רכישת ההשכלה)')])
            participant['steadyIncome'] = int(yesOrNo.get(row[titles.index('האם הנך בעל/ת הכנסה קבועה או מלגה?')]))
            participant['economicState']  = economicStates.get(row[titles.index('כיצד הנך תופס/ת את מצבך הסוציו-אקונומי?')])
            altruism = 0
            for i in [titles.index(' [אעזור לאדם זר שלא יודע את הדרך להגיע ליעד]'),
                      titles.index(' [אפרוט כסף עבור אדם זר]'),
                      titles.index(' [אתן כסף לצדקה]'),
                      titles.index(' [אתרום מוצרים/ בגדים לצדקה]'),
                      titles.index(' [אתנדב בארגון צדקה]'),
                      titles.index(' [אתרום דם]'),
                      titles.index(' [אעכב את המעלית עבור אדם זר כדי שיספיק לעלות]'),
                      titles.index(' [אתן למישהו שזקוק לכך לעקוף אותי בתור]'),
                      titles.index(' [אעזור לחבר ללימודים שאיני מכיר היטב במטלה שמתקשה בה ]'),
                      titles.index(' [אוותר על מקומי באוטובוס/רכבת בשביל זר שנאלץ לעמוד]')]:
                if row[i] == '5 (בסבירות גבוהה)':
                    altruism += 5
                    continue
                if row[i] == '1 (כלל לא סביר)':
                    altruism += 1
                    continue
                altruism += int(row[i])
            participant['altruism'] = altruism
            extroversion = 0
            introversion = 0
            extroversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [דברן]')])
            extroversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [אנרגטי]')])
            extroversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [מפגין התלהבות]')])
            extroversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [בעל אישיות אסרטיבית/החלטית]')])
            extroversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [חברותי ופתוח]')])
            introversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [מאופק]')])
            introversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [נוטה להיות שקט]')])
            introversion += agreement.get(row[titles.index('אני רואה עצמי כאדם... [עצור ומבוייש]')])
            #take mean of extra- and intra- version because number of questions are not equal
            participant['extroversion'] = extroversion / 5
            participant['introversion'] = introversion / 3
            participant['religion'] = religiousAffiliations.get(row[titles.index('השתייכות דתית')],"other")
    return

def main():
    participants = createParticipants()
    getSelfReportData(participants)
    getPreliminaryQuestionnaireData(participants)
    getVideoData(participants)
    getAudioData(participants)
    # variations of video data
    getVideoEmotionProbMean(VIDEO_DIR, 'VideoData', participants, 'videoProbMean')
    getVideoEmotionProbMean(VIDEO_BL_DIR, 'VideoBaselineData', participants,
                           'videoBLProbMean')
    getVideoAboveThresholdEmotionFreq(VIDEO_DIR, 'VideoData', participants,
                                      'videoThresholdFreq')
    getVideoAboveThresholdEmotionFreq(VIDEO_BL_DIR, 'VideoBaselineData',
                                      participants, 'videoBLThresholdFreq')
    getVideoEmotionFreqSkipBeginning(VIDEO_DIR, 'VideoData', participants,
                                     'videoFreqSkipBeg')
    getVideoEmotionFreqSkipBeginning(VIDEO_BL_DIR, 'VideoBaselineData',
                                     participants, 'videoBLFreqSkipBeg')
    getVideoFreqTopPercent(VIDEO_DIR, 'VideoData', participants, 'videoFreqTopPercent')
    getVideoFreqTopPercent(VIDEO_BL_DIR, 'VideoBaselineData', participants, 'videoBLFreqTopPercent')

    # remove invalid subjects
    removeInvalidParticipants(participants)

    # save dictionary
    with open(DATA_DIR + DICT_FILE, 'wb') as f:
        dump(participants, f)

    happy1 =[]
    happy2 = []
    sad1 = []
    sad2 = []
    neut1 = []
    neut2 = []
    pars = {
        "happy": [happy1, happy2],
        "sad": [sad1, sad2],
        "neutral": [neut1,neut2]
    }

    for participant in participants.values():
        stat = pars.get(participant['emotion'][0])
        stat[0].append(participant['ultimatumOffer'])
        stat[1].append(participant['trustOffer'])

    # mean & std
    print('happy ultimatum mean offer: {:0.3f}, std: {:0.3f}'.format(mean(happy1), stdev(happy1)))
    print('happy trust mean offer: {:0.3f}, std: {:0.3f}'.format(mean(happy2), stdev(happy2)))
    print('sad ultimatum mean offer: {:0.3f}, std: {:0.3f}'.format(mean(sad1), stdev(sad1)))
    print('sad trust mean offer: {:0.3f}, std: {:0.3f}'.format(mean(sad2), stdev(sad2)))
    print('neutral ultimatum mean offer: {:0.3f}, std: {:0.3f}'.format(mean(neut1), stdev(neut1)))
    print('neutral trust mean offer: {:0.3f}, std: {:0.3f}'.format(mean(neut2), stdev(neut2)))

    # t-test
    print('\nt-test:')
    # ultimatum
    [h,p] = ttest_ind(happy1,neut1)
    print('ultimatum: happy vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(sad1,neut1)
    print('ultimatum: sad vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(happy1,sad1)
    print('ultimatum: happy vs sad: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    # trust
    [h,p] = ttest_ind(happy2,neut2)
    print('trust: happy vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(sad2,neut2)
    print('trust: sad vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(happy2,sad2)
    print('trust: happy vs sad: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))

if __name__ == "__main__":
    main()
