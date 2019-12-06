'use strict'
const cote = require('cote')({statusLogsEnabled:false})
const u = require('elife-utils')
const level = require('level')
let lang
let languages
const LANG_AUTH = 'LANGUAGE'

function main() {
    console.log('loading default language......')
    loadLang()
    startMicroservice()
    registerWithCommMgr()
}

const commMgrClient = new cote.Requester({
    name: 'Elife-NW -> CommMgr',
    key: 'everlife-communication-svc',
})

function sendReply(msg, req) {
    req.type = 'reply'
    req.msg = msg
    commMgrClient.send(req, (err) => {
        if(err) u.showErr(err)
    })
}

const levelDBClient = new cote.Requester({
    name: 'Default Language skill Client',
    key: 'everlife-db-svc',
})


function loadLang(){
    levelDBClient.send({ type: 'get', key: LANG_AUTH }, (err, res) =>{
        if(err) console.log(err)
        else {
            if(res){
                console.log("Data loaded are"+JSON.parse(res))
                let data = JSON.parse(res) 
                languages = {}
                languages[lang] = data['prefer_lang']
                lang = languages      
            }
        }
    })
}

let msKey = 'everlife-defaultLang'

function registerWithCommMgr() {
    commMgrClient.send({
        type: 'register-msg-handler',
        mskey: msKey,
        mstype: 'msg',
        mshelp: [
            { cmd: '/set_language', txt: 'For setting user language ' } 
        ],
    }, (err) => {
        if(err) u.showErr(err)
    })
}

function startMicroservice() {
    const svc = new cote.Responder({
        name: 'Everlife-NW Service',
        key: msKey,    
    })
    let languageDefault = {}
    let islanguage = false
    svc.on('msg', (req, cb) => {
        if(!req.msg) return cb()
        const msg = req.msg.trim()
        if(msg.startsWith('/set_language')) {
            cb(null, true)
            if(!lang){
                lang = {}
                sendReply("Enter your preferred language 1.Spanish(sp) 2.English(en)",req)
                islanguage = true
            }else{
                cb(null,true)
                sendReply('You have already set your preferred language.',req)
            }
        }else if(islanguage && !languageDefault.prefer_lang){
            languageDefault['prefer_lang'] = req.msg
            cb(null,true)
            islanguage = false
            sendReply('Setting your preferred language......',req)
            languages = lang
            languageDefault['prefer_lang'] = lang.prefer_lang
            levelDBClient.send({ type: 'put', key: LANG_AUTH, val: JSON.stringify(languageDefault) }, (err) => {
                if(err) console.log(err)
                else{
                    console.log("DATA STORED")
                    console.log(languageDefault)
                }
            })
            sendReply("Language set",req) 
            cb(null,true)    
        }else {
            cb()
        }
    })
} main()
