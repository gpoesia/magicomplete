import * as express from 'express';
import * as bodyParser from 'body-parser';
import * as Fs from 'fs';
import mongoose from 'mongoose';
import proxy from 'express-http-proxy';

const mongoPort = process.env['MONGO_PORT'] || 27017;
const magicompletePort = process.env['MAGICOMPLETE_PORT'] || 5000;

mongoose.connect(`mongodb://localhost:{mongoPort}/autocomplete`,
                 { useNewUrlParser: true, useUnifiedTopology: true });

const EventLog = mongoose.model('EventLog', {
  session: String,
  target: String,
  timestamp: Date,
  setting: Number,
  events: Array,
});

const dataset = JSON.parse(
  Fs.readFileSync('./user-study.json', { encoding: 'utf8' }));

console.log('Loaded dataset with', dataset.length, 'examples.');

const app = express();
app.use(bodyParser.json({ limit: '5mb' }));

app.get('/dataset', (req, res) => {
  res.json(dataset);
});

app.post('/save-events', (req, res) => {
  const log = new EventLog({
    session: req.body.session,
    target: req.body.target,
    timestamp: new Date(),
    events: req.body.events,
    setting: req.body.setting,
  });

  log.save().then(() => res.send('OK'));
});

app.get('/keywords', proxy(`localhost:${magicompletePort}/keywords`));
app.get('/complete', proxy(`localhost:${magicompletePort}/complete`));

const port = process.env.port || 3333;
const server = app.listen(port, () => {
  console.log('Listening at http://localhost:' + port);
});
server.on('error', console.error);
