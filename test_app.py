"""Synthetic integration checks; these do not evaluate model accuracy."""
import io
import os
from pathlib import Path
import tempfile
import unittest

temporary = tempfile.TemporaryDirectory(prefix='clingastrovqa-test-')
os.environ['CLINGASTRO_DEMO'] = '1'
os.environ['CLINGASTRO_RUNS_DIR'] = temporary.name
os.environ['MPLBACKEND'] = 'Agg'

from PIL import Image
from fastapi.testclient import TestClient
from clingastrovqa.main import app
from clingastrovqa import config, main
from clingastrovqa.findings import fuse

class ApplicationTests(unittest.TestCase):
    def test_upload_to_download(self):
        image = io.BytesIO()
        Image.new('RGB', (128, 128), (180, 80, 80)).save(image, 'PNG')
        with TestClient(app) as client:
            health = client.get('/api/health').json()
            self.assertEqual(health['mode'], 'demo')
            self.assertTrue(health['question_bank_loaded'])
            response = client.post('/api/analyze', files={'file':('synthetic.png', image.getvalue(), 'image/png')})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(len(result['vqa']), 19)
            self.assertEqual(result['mode'], 'demo')
            self.assertIn('SYNTHETIC DEMO', result['disclaimer'])
            pdf = client.get(result['report']['pdf'])
            self.assertEqual(pdf.status_code, 200)
            self.assertTrue(pdf.content.startswith(b'%PDF'))
            html = client.get(result['report']['html'])
            self.assertIn('SYNTHETIC DEMO', html.text)
            self.assertEqual(client.get(result['report']['json']).status_code, 200)
            for url in result['figures'].values():
                self.assertEqual(client.get(url).status_code, 200)
            self.assertEqual(client.get('/api/runs/not-a-run').status_code, 404)
            self.assertEqual(client.post('/api/analyze', files={'file':('bad.png', b'not an image', 'image/png')}).status_code, 400)

    def test_live_requires_models(self):
        with TestClient(app) as client:
            config.DEMO_MODE = False
            try:
                self.assertFalse(main._ready())
                response = client.post('/api/analyze', files={'file':('image.png', b'anything', 'image/png')})
                self.assertEqual(response.status_code, 503)
            finally:
                config.DEMO_MODE = True

    def test_unsupported_answer_does_not_become_finding(self):
        result = fuse('synthetic', None, {'abnormalities':{'answer':'polyp','in_coverage':False}}, None, None)
        self.assertEqual(result.abnormalities, [])
        self.assertIsNone(result.has_finding)
        self.assertIsNone(result.procedure)
        self.assertEqual(result.vqa['abnormalities']['answer'], 'polyp')

if __name__ == '__main__':
    unittest.main()
