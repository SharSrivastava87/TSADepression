// src/components/Header.tsx
import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, Button } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import { Link } from 'react-router-dom';

const Header = () => (
    // Color Picker
  <AppBar position="static" sx={{ backgroundColor: '#ff6d00' }}>
    <Toolbar>
      <Typography variant="h6" sx={{ flexGrow: 1, color: '#FFFFFF' }}>
        Logo
      </Typography>
      <Button component={Link} to="/" color="inherit" sx={{ margin: '0 10px' }}>
        Home
      </Button>
      <Button component={Link} to="/getting-started" color="inherit" sx={{ margin: '0 10px' }}>
        Getting Started
      </Button>
      <Button component={Link} to="/dashboard" color="inherit" sx={{ margin: '0 10px' }}>
        Dashboard
      </Button>
      <Button component={Link} to="/login" color="inherit" sx={{ margin: '0 10px' }}>
        Log In
      </Button>
      <Button component={Link} to="/sign-up" color="inherit" sx={{ margin: '0 10px' }}>
        Sign Up
      </Button>
      <IconButton edge="end" color="inherit">
        <SettingsIcon sx={{ color: '#FFFFFF' }} />
      </IconButton>
    </Toolbar>
  </AppBar>
);

export default Header;
