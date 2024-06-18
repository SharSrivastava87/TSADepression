// src/App.tsx
import React from 'react';
import { ThemeProvider, CssBaseline, Box } from '@mui/material';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import MainContent from './components/MainContent';
import theme from './theme';
import GettingStarted from './components/GettingStarted';
import Dashboard from './components/Dashboard';
import Login from './components/Login'; // Import Login component
import SignUp from './components/SignUp'; // Import SignUp component

const App = () => (
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <Router>
      <Header />
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
        }}
      >
        <Routes>
          <Route path="/" element={<MainContent />} />
          <Route path="/getting-started" element={<GettingStarted />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/login" element={<Login />} /> {/* Add Login route */}
          <Route path="/sign-up" element={<SignUp />} /> {/* Add SignUp route */}
        </Routes>
      </Box>
    </Router>
  </ThemeProvider>
);

export default App;
