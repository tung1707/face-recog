from flask import Blueprint
from controllers.RecognitionController import home,addprsn,addprsn_submit,fr_page,video_feed,vfdataset_page,vidfeed_dataset,loadData,countTodayScan,train_classifier

recognition = Blueprint('recognition',__name__)
recognition.route('/',methods=['GET'])(home)
recognition.route('/addprsn')(addprsn)
recognition.route('/addprsn_submit', methods=['POST'])(addprsn_submit)
recognition.route('/fr_page')(fr_page)
recognition.route('/video_feed')(video_feed)
recognition.route('/vfdataset_page/<prs>')(vfdataset_page)
recognition.route('/vidfeed_dataset/<nbr>')(vidfeed_dataset)
recognition.route('/loadData', methods=['GET','POST']) (loadData) #Load data điểm danh lên trang nhận diện 
recognition.route('/countTodayScan')(countTodayScan)
recognition.route('/train_classifier/<nbr>')(train_classifier)
