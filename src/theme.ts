// src/theme.ts
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1515bc', // Customize your primary color
    },
    secondary: {
      main: '#FF6D00', // Customize your secondary color
    },
    background: {
      default: '#1E1E2F', // Background color
      paper: '#2C2C3E', // Paper background color
    },
    text: {
      primary: '#FFFFFF', // Primary text color
      secondary: '#B0B0B0', // Secondary text color
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif', // Customize your font family
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
    },
    body1: {
      fontSize: '1rem',
      color: '#B0B0B0',
    },
  },
});

export default theme;
