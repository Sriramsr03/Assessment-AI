import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { StudentPage } from './pages/StudentPage.jsx';
import { TeacherPage } from './pages/TeacherPage.jsx';
import { TeacherReportPage } from './pages/TeacherReportPage.jsx';
import './App.css';

export function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <main className="main">
          <Routes>
            <Route path="/" element={<TeacherPage />} />
            <Route path="/assessment" element={<StudentPage />} />
            <Route path="/report" element={<TeacherReportPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
